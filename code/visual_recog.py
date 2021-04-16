import numpy as np
import threading
import queue
import imageio
import os, time
import math
import visual_words
import multiprocessing
import skimage
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle=True)
	dictionary = np.load("../results/dictionary.npy")
	# print(train_data.files)
	# print(train_data['image_names'].shape)
	# print(train_data['labels'])

	# get the laynum parameter
	layer_num = 3
	num_image = train_data['labels'].shape[0]
	# Get the label array
	labels = train_data['labels']
	labels = np.array(labels)
	# Get K, which is the cluster number
	K, _ = dictionary.shape
	# The feature container
	features = []

	for i in trange(int(num_image / num_workers)):
		# Multithreading
		pool = multiprocessing.Pool(num_workers)
		param = []
		for j in range(num_workers):
			param.append((train_data['image_names'][i * num_workers + j][0], dictionary, layer_num, K))
		results = pool.map(get_image_feature, param)

		features.extend(results)

	features = np.vstack(features)
	np.savez('../results/trained_system.npz', dictionary=dictionary, features=features, labels=labels,
			 SPM_layer_num=layer_num)


def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	trained_system = np.load("../results/trained_system.npz")
	# print(test_data.files)
	conf = np.zeros((8, 8))
	iter = test_data['labels'].shape[0]
	actual_labels = test_data['labels']
	features = trained_system['features']
	labels = trained_system['labels']
	dictionary = trained_system['dictionary']
	layer_num = trained_system['SPM_layer_num']
	K, _ = dictionary.shape
	wrong_prediction = []

	for i in trange(iter):
		file_path = test_data['image_names'][i][0]
		feature = get_image_feature(file_path, dictionary, layer_num, K)
		similarity = distance_to_set(feature, features)
		index = np.argmax(similarity)
		predicted_label = labels[index]
		actual_label = actual_labels[i]
		if predicted_label != actual_label:
			wrong_prediction.append((file_path, actual_label, predicted_label))
		conf[actual_label, predicted_label] += 1

	accuracy = np.trace(conf) / np.sum(conf)
	np.save("../results/wrong_prediction.npy", wrong_prediction)
	return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K*(4^layer_num-1)/3))
	'''

	# file_path, dictionary, layer_num, K = args
	image = skimage.io.imread("../data/" + file_path)
	wordmap = visual_words.get_visual_words(image, dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)

	return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    dist_dict = np.minimum(histograms, word_hist)
    similarity = np.sum(dist_dict, axis=1)
    return similarity


def get_feature_from_wordmap(wordmap, dict_size):
    '''
	Extracts the spatial pyramid matching feature.
	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps
	[output]
	* feature: numpy.ndarray of shape (K*(4^layer_num-1)/3))
	'''

    H, W = wordmap.shape
    flatten = wordmap.reshape(-1)
    bin = np.arange(dict_size + 1)
    hist, bin_return = np.histogram(flatten, bins=bin, density=True)
    # plt.bar(bin_return[:-1], hist, width=1)
    # plt.xlim(min(bin_return), max(bin_return))
    # plt.show()
    hist = hist.reshape((1, dict_size))

    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
	"""
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	"""

	H, W = wordmap.shape
	weights = []
	for i in range(layer_num):
		if i == 0 or i == 1:
			weights.append(2.0**(-(layer_num-1)))
		else:
			weights.append(2.0**(i-layer_num))

	hist_all = []
	for i in range(layer_num - 1, -1, -1):
		section = np.power(2, i)
		weight = weights[i]
		height = int(H / section)
		width = int(W / section)

		for a in range(section):
			for b in range(section):
				img = wordmap[height * a: height * (a + 1), width * b:width * (b + 1)]
				hist = get_feature_from_wordmap(img, dict_size)  # 1 * K
				hist_all.append(hist)

	hist_all = np.hstack(hist_all)

	return hist_all


if __name__ == '__main__':
    # get_feature_from_wordmap()
    evaluate_recognition_system()
# pass
