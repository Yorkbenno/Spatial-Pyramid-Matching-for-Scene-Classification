import numpy as np
import scipy.ndimage
import os, time
import skimage


def extract_deep_feature(x, vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	x = skimage.transform.resize(x, (224, 224))
	c = x.shape[2]
	if c == 1:
		x = np.matlib.repmat(x, 1, 1, 3)
	if c == 4:  # Weird, but found this error while processing
		x = x[:, :, 0:3]
	for i in range(3):
		x[:, :, i] = (x[:, :, i] - mean[i]) / std[i]

	L = len(vgg16_weights)
	count = 0
	for l in range(L):
		params = vgg16_weights[l]
		if params[0] == 'conv2d':
			x = multichannel_conv2d(x, params[1], params[2])
		elif params[0] == 'relu':
			x = relu(x)
		elif params[0] == 'maxpool2d':
			x = max_pool2d(x, params[1])
		elif params[0] == 'linear':
			x = linear(x, params[1], params[2])
			count += 1
			if count == 2:
				break

	feat = x.reshape(1, np.shape(x)[0])
	return feat


def multichannel_conv2d(x, weight, bias):
    '''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''

    H, W, input_dim = x.shape
    output_dim, input_dim, kernel_size, kernel_size = weight.shape
    feat = []

    for i in range(output_dim):
        value = np.zeros((H, W, 1))

        for j in range(input_dim):
            filter = weight[i, j, :]
            filter = np.flip(filter)
            channel = x[:, :, j]
            value[:, :, 0] += scipy.ndimage.convolve(channel, filter, mode='constant')

        value += bias[i]
        feat.append(value)

    feat = np.dstack(feat)

    return feat


def relu(x):
    '''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''

    y = np.maximum(0, x)
    return y


def max_pool2d(x, size):
    '''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''

    H, W, input_dim = x.shape
    H_after = int(H / size)
    W_after = int(W / size)

    y = x.reshape(H_after, size, W_after, size, input_dim).max(axis=(1, 3))
    return y


def linear(x, W, b):
    '''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
    # special case in first layer
    y = W @ x.T.flatten() + b
    return y
