#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains various functions for layer construction.

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################

    N, H, W, C = x.shape
    F, HH, WW, _ = w.shape
    H_filter = 1 + (H + 2 * pad - HH) // stride
    W_filter = 1 + (W + 2 * pad - WW) // stride
    print(H_filter)
    print(W_filter)

    A = np.zeros((N, F, HH, WW))

    npad = ((0, 0), (pad, pad), (pad, pad), (0, 0))

    # Pad the input with zeros
    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    #print(x.shape)
    for i in range(N):  # ith example
        for j in range(F):  # jth filter
            # Convolve this filter over windows
            for k in range(H_filter):
                hs = k * stride
                for l in range(W_filter):
                    ws = l * stride
                    # Window we want to apply the respective jth filter over (C, HH, WW)
                    window = x[i, hs:hs + HH, ws:ws + WW, :]
                    # Convolve
                    A[i, k, l, j] = np.sum(window * w[j:,:,:]) + b[j]
    return A

    #raise NotImplementedError


def conv2d_backward(dout, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you +10 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################

    N, H, W, C = x.shape
    F, HH, WW, _ = w.shape
    H_filter = dout.shape[1]
    W_filter = dout.shape[2]

    # Initialize matrices for gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    padded = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    padded_dx = np.pad(dx, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # Backpropagate dout through each input patch and each convolution filter
    for i in xrange(N):  # ith example
        for j in xrange(F):  # jth filter
            # Convolve this filter over windows
            for k in xrange(Hp):
                hs = k * stride
                for l in xrange(Wp):
                    ws = l * stride

                    # Window we applies the respective jth filter over (C, HH, WW)
                    window = padded[i, :, hs:hs + HH, ws:ws + WW]

                    # Compute gradient of out[i, j, k, l] = np.sum(window*w[j]) + b[j]
                    db[j] += dout[i, j, k, l]
                    dw[j] += window * dout[i, j, k, l]
                    padded_dx[i, :, hs:hs + HH, ws:ws + WW] += w[j] * dout[i, j, k, l]

                    # "Unpad"
    dx = padded_dx[:, :, pad:pad + H, pad:pad + W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx[:, :, pad:-pad, pad:-pad], dw, db


    #raise NotImplementedError

