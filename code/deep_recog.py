import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os, time
import torch
import skimage.transform
import torchvision.transforms
from tqdm import tqdm, trange
import util
import scipy.spatial
import network_layers


def build_recognition_system(vgg16, num_workers=2):
    '''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''

    train_data = np.load("../data/train_data.npz", allow_pickle=True)
    num_image = train_data['labels'].shape[0]
    # Get the label array
    labels = train_data['labels']
    labels = np.array(labels)

    features = []

    for i in trange(int(num_image / num_workers)):
        # Multithreading
        pool = multiprocessing.Pool(num_workers)
        param = []
        for j in range(num_workers):
            param.append((i * num_workers + j, train_data['image_names'][i * num_workers + j][0], vgg16))
        results = pool.map(get_image_feature, param)

        features.extend(results)

    features = np.vstack(features)

    np.savez('../results/trained_system_deep2.npz', features=features, labels=labels)


def evaluate_recognition_system(vgg16, num_workers=2):
    '''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

    test_data = np.load("../data/test_data.npz", allow_pickle=True)
    trained_system_deep = np.load("../results/trained_system_deep2.npz")

    conf = np.zeros((8, 8))
    iter = test_data['labels'].shape[0]
    actual_labels = test_data['labels']
    features = trained_system_deep['features']
    labels = trained_system_deep['labels']


    for i in trange(int(iter / num_workers)):
        # Multithreading
        pool = multiprocessing.Pool(num_workers)
        param = []
        for j in range(num_workers):
            param.append((i * num_workers + j, test_data['image_names'][i * num_workers + j][0], vgg16))
        results = pool.map(get_image_feature, param)
        for num, k in enumerate(results):
            similarity = distance_to_set(k, features)
            index = np.argmax(similarity)
            predicted_label = labels[index]
            actual_label = actual_labels[i * num_workers + num]
            conf[actual_label, predicted_label] += 1

    accuracy = np.trace(conf) / np.sum(conf)
    np.save("../results/conf_deep.npy", conf)

    return conf, accuracy


def preprocess_image(image):
    '''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

    image = skimage.transform.resize(image, (224, 224))
    c = image.shape[2]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if c == 1:
        image = np.matlib.repmat(image, 1, 1, 3)
    if c == 4:  # Weird, but found this error while processing
        image = image[:, :, 0:3]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=mean, std=std)])
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def get_image_feature(args):
    '''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
	[saved]
	* feat: evaluated deep feature
	'''

    # i, image_path, vgg16 = args
    # image = skimage.io.imread("../data/" + image_path)
    # image = preprocess_image(image)
    #
    # linear1 = torch.nn.Sequential(*list(vgg16.children())[0])
    # adaptivepool = list(vgg16.children())[1]
    # linear2 = torch.nn.Sequential(*list(vgg16.children())[2][:4])
    #
    # feat = linear2(adaptivepool(linear1(image)).flatten())
    # feat = feat.detach().numpy().reshape(1, 4096)
    # np.save('../results/feat.npy', feat)
    #
    # If we use our self-built model, use the below code
    i, image_path, vgg16 = args
    image = skimage.io.imread("../data/" + image_path)
    feat = network_layers.extract_deep_feature(image, vgg16)
    np.save('../results/feat.npy', feat)

    return feat


def distance_to_set(feature, train_features):
    '''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

    distance = scipy.spatial.distance.cdist(feature, train_features, metric='euclidean')
    dist = -1 * distance
    return dist
