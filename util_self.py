# IMPORTS

import itertools
import os
import subprocess
import time
import gc
import keras.utils as ku

from nearpy import Engine
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from lazy_greedy_self import FacilityLocation, lazy_greedy_heap
import scipy.spatial

# MOST GENERAL PARS AND CONTAINERS

SEED = 42
EPS = 1E-8
PLOT_NAMES = ['lr', 'data_loss', 'epoch_loss', 'test_loss'] 
datasets = ['mnist']

# LODING THE DATA (STARTED FROM MNIST ONLY)

def load_dataset(dataset):

	'''
	dataset - string, e.g. 'mnist'

	'''
	if dataset == 'mnist':
		from keras.datasets import mnist
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train = (X_train / 255).reshape(-1,784)
		X_test = (X_test / 255).reshape(-1,784)

		#y_train_old = y_train
		#y_test_old = y_test

		y_train = ku.to_categorical(y_train, dtype ="uint8") 
		y_test = ku.to_categorical(y_test, dtype ="uint8") 
		print('MNIST captured from Keras', flush = True)
		return X_train, y_train, X_test, y_test

	elif dataset not in datasets:
		print(f'Unknown data requirement {dataset}')


# SIMILARITY COMPUTATION

def similarity(X, metric):

    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]

    '''
    print('X_shape in dists:', X.shape)
    start = time.time()
    dists = sklearn.metrics.pairwise_distances(X, metric=metric, n_jobs=-1)
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists # turning to similarities
    elif metric == 'euclidean' or metric == 'l1':
        print('dists_shape: ',dists.shape)
        m = np.max(dists) # metrix from the paper, max of divergences
        S = m - dists # turning to similarities
    else:
        raise ValueError(f'unknown metric: {metric}')

    return S, elapsed


def greedy_merge(X, y , B, part_num, metric, smtk = 0, stoch_greedy = False):

	# Still dunno the functionality of get_orders_and_weights

	'''
	X - array of examp-features
	part_num - int, number of parts we want to split the data in

	'''
	N = len(X)
	indices = list(range(N))
	part_size = int(np.ceil(N/part_num))
	part_indices = [indices[slice(i * part_size, min((i+1)* part_size, N))] for i in range(part_num)]
	print(f'GreeDi with {part_num} parts, searching for B elements...', flush = True)

	order_mg, cluster_sizes_all, _, _, ordering_time, similarity_time, F_val = zip(*map(
		lambda p: get_orders_and_weights(
			int(B/2),
			X[ part_indices[p],: ],
			metric, 
			p + 1,
			stoch_greedy,
			y[part_indices[p]]),
			np.arange(part_num)
		))

	collected = gc.collect() # taking care of memory cleaning
	print(f'Garbage collector collected : {collected}', flush = True)

	# Stage 1 of GreeDi algorithm

	order_mg_all = np.array(order_mg_all, dtype=np.int32) # change the type of the data in vector
	cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32) # change the type for clusters vector
	order_mg = order_mg_all.flatten(order='F') # Fortran col order
	weights_mg = cluster_sizes_all.flatten(order='F') # using flattened cluster sizes as weights
	print(f'GreeDi stage 1: found {len(order_mg)} elements in {np.max(ordering_time)} sec', flush = True)

	# Stage 2 of GreeDi algorithm

	order, weights, order_sz, weights_sz, ordering_time_merge, similarity_time_merge = get_orders_and_weights(
		B,
		X[order_mg, :],
		metric,
		smtk,
		0,
		stoch_greedy,
		y[order_mg],
		weights_mg)

	print(f'Weights: {weights}')

	total_ordering_time = np.max(similarity_time) + ordering_time_merge
	total_similarity_time = np.max(similarity_time) + similarity_time_merge

	print(f'GreeDi stage 2: found {len(order)} elements in: {total_ordering_time + total_similarity_time} sec', flush=True)

	vals = order, weights, order_sz, weights_sz, total_ordering_time, total_similarity_time

	return vals


def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, outdir='.'):
    
    '''
    Input args:

    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns:

    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''

    N = X.shape[0] # keep the X shape

    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class

    classes = np.arange(0,10)
    C = len(classes)  # number of classes
    print('C = ', C)

    # equalizing the size of class representatives by mask if needed
    if equal_num:
        class_nums = [sum(y == ku.to_categorical(c,num_classes = 10)) for c in classes]
        print('class_nums : ', class_nums)
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        print(f'classes = {classes}')
        print(f'observe: {y == ku.to_categorical(1, num_classes = 10)}')
        num_per_class = [np.ceil(0.1 * B).astype(np.int64) for i in classes] #np.int32(np.ceil(np.divide([np.sum(y == ku.to_categorical(i, num_classes = 10)) for i in classes], N) * B))
        print('Not equal_num, which may be OK if intended')



    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: facility_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = [0.1 for i in range(10)]#np.divide([np.sum(y == i) for i in classes], N) ####################33
        props = np.rint(class_ratios / np.min(class_ratios))
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(props)}')

    order_mg_all = np.array(order_mg_all)
    cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
        for c in classes:
            ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = [] # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals



def facility_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None):

    class_indices = []
    c_plus = ku.to_categorical(c, num_classes = 10).astype(np.int64)
    #print(c_plus)
    y_plus = y.reshape(len(X), - 1)
    #print(y_plus[0, :])   

    #print('-'*50)
    #print(y_plus.shape)
    #print(c_plus.shape)
    #print(X.shape)
    #print('-'*50)

    for i in range(len(y_plus)):
        if np.all(y_plus[i] == c_plus):
            class_indices.append(i)

    print(len(class_indices))

    class_indices = np.asarray(class_indices).astype(np.int64)

    
    

    S, S_time = similarity(X[class_indices,:], metric=metric)

    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S,
        num_per_class,
        c,
        smtk, 
        no, 
        stoch_greedy, 
        weights)

    return class_indices[order], cluster_sz, greedy_time, S_time



def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None):

    '''

    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select
    - c: specific class we work with (out of the label set)

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point

    '''


    N = S.shape[0] # keep the shape
    no = smtk if no == 0 else no # set the flags

    if smtk > 0: # didn't get this part, some outer syntax
        print(f'Calculating ordering with SMTK... part size: {len(S)}, B: {B}', flush=True)
        np.save(f'/tmp/{no}/{smtk}-{c}', S)
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy /tmp/{no}/{smtk}-{c}.'
                f'npy -pnpv -porder -ptime'.split())
        else:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                             -flnpy /tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'.split())

        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        greedy_time = float(s[s.find('CPU') + 4 : s.find('s (User')])
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
    else:
        V = list(range(N))
        start = time.time()
        F = FacilityLocation(S, V) # make use of lazy_greedy_self.py functionality
        order, _ = lazy_greedy_heap(F, V, B)
        greedy_time = time.time() - start
        F_val = 0

    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]

    collected = gc.collect() # caring about memory again
    return order, sz, greedy_time, F_val

# saving utils

def save_all_orders_and_weights(folder, X, metric='l2', stoch_greedy=False, y=None, equal_num=False, outdir='.'):
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class



    classes = np.arange(0,10)
    C = len(classes)  # number of classes
    # assert np.array_equal(classes, np.arange(C))
    # assert B % C == 0
    print(f'number_of_classes: {C}')
    class_nums = [sum(y == ku.to_categorical(c, num_classes = C)) for c in classes]
    print(class_nums)

    ##################

    class_indices = []
    c_plus = ku.to_categorical(c, num_classes = C).astype(np.int64)
    y_plus = y.reshape(len(X), - 1)
    print(y_plus[0, :])   

    for i in range(len(y_plus)):
        if np.all(y_plus[i] == c_plus):
            class_indices.append(i)

    class_indices = np.asarray(class_indices).astype(np.int64)

    #######################

    tmp_path = '/tmp'
    no, smtk = 2, 2

    def greedy(B, c):
        print('Computing facility location submodular order...')
        print(f'Calculating ordering with SMTK... part size: {class_nums[c]}, B: {B}', flush=True)
        command = f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                                 -flnpy {tmp_path}/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'
        if stoch_greedy:
            command += f' -stochastic-greedy -sg-epsilon {.9}'

        p = subprocess.check_output(command.split())
        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        order = np.asarray(order, dtype=np.int64)
        greedy_time = float(s[s.find('CPU') + 4: s.find('s (User')])
        print(f'FL greedy time: {greedy_time}', flush=True)
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
        print(f'===========> f(Solution) = {F_val}')
        print('time (sec) for computing facility location:', greedy_time, flush=True)
        return order, greedy_time, F_val

    def get_subset_sizes(B, equal_num):
        if equal_num:
            # class_nums = [sum(y == c) for c in classes]
            num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
            minority = class_nums < np.ceil(B / C)
            if sum(minority) > 0:
                extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
                for c in classes[~minority]:
                    num_per_class[c] += int(np.ceil(extra / sum(minority)))
        else:
            num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))

        return num_per_class

    def merge_orders(order_mg_all, weights_mg_all, equal_num):
        order_mg, weights_mg = [], []
        if equal_num:
            props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
        else:
            # merging imbalanced classes
            print('Classes:', С)
            class_ratios = np.divide([np.sum(y == ku.to_categorical(i)) for i in classes], N) ################
            props = np.rint(class_ratios / np.min(class_ratios))
            print(f'Selecting with ratios {np.array(class_ratios)}')
            print(f'Class proportions {np.array(props)}')

        order_mg_all = np.array(order_mg_all)
        weights_mg_all = np.array(weights_mg_all)
        for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
            for c in classes:
                ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
                order_mg = np.append(order_mg, order_mg_all[c][ndx])
                weights_mg = np.append(weights_mg, weights_mg_all[c][ndx])
        order_mg = np.array(order_mg, dtype=np.int32)
        weights_mg = np.array(weights_mg, dtype=np.float)
        return order_mg, weights_mg

    def calculate_weights(order, c):
        weight = np.zeros(len(order), dtype=np.float64)
        center = np.argmax(D[str(c)][:, order], axis=1)
        for i in range(len(order)):
            weight[i] = np.sum(center == i)
        return weight

    D, m = {}, 0
    similarity_times, max_similarity = [], []
    for c in classes:
        print(f'Computing distances for class {c}...')
        time.sleep(.1)
        start = time.time()
        if metric in ['', 'l2', 'l1']:
            dists = sklearn.metrics.pairwise_distances(X[class_indices[c]], metric=metric, n_jobs=1)
        else:
            p = float(metric)
            dim = class_nums[c]
            dists = np.zeros((dim, dim))
            for i in range(dim):
                dists[i,:] = np.power(np.sum(np.power(np.abs(X[class_indices[c][i]] - X[class_indices[c]]), p), axis=1), 1./p)
                # for j in range(i+1, dim):
                #     dists[i,j] = np.power(np.sum(np.power(np.abs(X[class_indices[c][i]] - X[class_indices[c][j]]), p)), 1./p)
            # dists[np.triu_indices(dim, 1)] = d
            # dists = dists.T + dists
        similarity_times.append(time.time() - start)
        print(f'similarity times: {similarity_times}')
        print('Computing max')
        m = np.max(dists)
        print(f'max: {m}')
        S = m - dists
        np.save(f'{tmp_path}/{no}/{smtk}-{c}', S)
        D[str(c)] = S
        max_similarity.append(m)

    # Ordering all the data with greedy
    print(f'Greedy: selecting {class_nums} elements')
    # order_in_class, greedy_times, F_vals = zip(*map(lambda c: greedy(class_nums[c], c), classes))
    # order_all = [class_indices[c][order_in_class[c]] for c in classes]

    for subset_size in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # for subset_size in [0.9, 1.0]:
        B = int(N * subset_size)
        num_per_class = get_subset_sizes(B, equal_num)

        # Note: for marginal gains
        order_in_class, greedy_times, F_vals = zip(*map(lambda c: greedy(num_per_class[c], c), classes))
        order_all = [class_indices[c][order_in_class[c]] for c in classes]
        #####

        weights = [calculate_weights(order_in_class[c][:num_per_class[c]], c) for c in classes]
        order_subset = [order_all[c][:num_per_class[c]] for c in classes]
        order_merge, weights_merge = merge_orders(order_subset, weights, equal_num)
        F_vals = np.divide(F_vals, class_nums)

        folder = '/tmp/covtype'
        print(f'saving to {folder}_{subset_size}_{metric}_w.npz')
        np.savez(f'{folder}_{subset_size}_{metric}_w', order=order_merge, weight=weights_merge,
                 order_time=greedy_times, similarity_time=similarity_times, F_vals=F_vals, max_dist=m)
    # end for on subset sizes
    # return vals