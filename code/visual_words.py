import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random


def extract_filter_responses(image):
    """
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    # ----- TODO -----

    H, W, c = image.shape

    if isinstance(image[0, 0, 0], int):
        image = image.astype('float') / 255
    elif np.amax(image) > 1:
        image = image.astype('float') / 255

    if c == 1:
        image = np.matlib.repmat(image, 1, 1, 3)
    if c == 4:  # Weird, but found this error while processing
        image = image[:, :, 0:3]

    image = skimage.color.rgb2lab(image)
    scales = [1, 2, 4, 8, 8 * np.sqrt(2)]
    filter_responses = np.zeros((H, W, 60))

    for i in range(5):
        for j in range(3):
            filter_responses[:, :, 12 * i + j] = scipy.ndimage.gaussian_filter(image[:, :, j], sigma=scales[i])
            filter_responses[:, :, 12 * i + 3 + j] = scipy.ndimage.gaussian_laplace(image[:, :, j],
                                                                                    sigma=scales[i])
            filter_responses[:, :, 12 * i + 6 + j] = scipy.ndimage.gaussian_filter(image[:, :, j],
                                                                                   sigma=scales[i],
                                                                                   order=[0, 1])
            filter_responses[:, :, 12 * i + 9 + j] = scipy.ndimage.gaussian_filter(image[:, :, j],
                                                                                   sigma=scales[i],
                                                                                   order=[1, 0])
    return filter_responses


def get_visual_words(image, dictionary):
    """
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    # ----- TODO -----

    filter_responses = extract_filter_responses(image)
    H, W, channel = filter_responses.shape
    k, channel = dictionary.shape
    wordmap = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            pixel = filter_responses[i, j, :]
            pixel = pixel.reshape(1, channel)
            distance = scipy.spatial.distance.cdist(dictionary, pixel, metric='euclidean')
            # shape is K * 1
            wordmap[i, j] = np.argmin(distance)

    return wordmap


def compute_dictionary_one_image(args):
    """
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    """

    i, alpha, image_path = args
    # ----- TODO -----

    image = skimage.io.imread("../data/" + image_path[i][0])
    filter_responses = extract_filter_responses(image)

    H, W, channel = filter_responses.shape
    flatten = filter_responses.reshape((H * W, channel))
    index = np.random.choice(range(H * W), size=alpha)
    flatten = flatten[index, :]

    if not os.path.exists("../results"):
        os.mkdir("../results")
    np.save('../results/sampled_response.npy', flatten)

    return flatten


def compute_dictionary(num_workers=2):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    """

    train_data = np.load("../data/train_data.npz", allow_pickle=True)
    # ----- TODO -----

    # print(train_data.files)
    # print(np.shape(train_data['image_names'])[0])
    # print(train_data['image_names'][0][0])
    samples_num = train_data['image_names'].shape[0]  # 1440 samples
    alpha = 250
    K = 200

    filter_response = []
    for i in range(int(samples_num / num_workers)):
        pool = multiprocessing.Pool(num_workers)
        param = []
        for j in range(num_workers):
            param.append((i * num_workers + j, alpha, train_data['image_names']))
        results = pool.map(compute_dictionary_one_image, param)

        filter_response.extend(results)

    filter_response = np.vstack(filter_response)

    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-1).fit(filter_response)
    dictionary = kmeans.cluster_centers_
    np.save('../results/dictionary.npy', dictionary)

    return dictionary


if __name__ == '__main__':
    compute_dictionary()
