import numpy as np
from random import shuffle

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros(W.shape).astype('float')  # initialize the gradient as zero

    # compute the loss and the gradient
    num_dims = W.shape[0]
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        scores_exp = np.exp(scores)
        scores = scores_exp/np.sum(scores_exp)
        correct_class_score = scores[y[i]]
        for d in range(num_dims):
            for k in range(num_classes):
                if k == y[i]:
                    dW[d, k] += X.T[d, i] * (scores[k] - 1)
                else:
                    dW[d, k] += X.T[d, i] * scores[k]
        loss += -np.log(correct_class_score)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    dW /= num_train
    dW += reg * W


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    loss = 0.0
    num_dims = W.shape[0]
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros_like(W)

    scores = np.dot(X,W)
    scores_exp = np.exp(scores)
    score_sum = np.sum(scores_exp,axis=1)
    score_sum = (np.ones(scores.shape).T*score_sum).T
    scores = scores_exp/score_sum
    correct_score = scores[range(num_train),y]
    correct_score = np.log(correct_score)
    loss = np.sum(-correct_score)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)

    dscores = scores
    dscores[range(num_train), y] -= 1
    dW = np.dot(X.T, dscores)
    dW /= num_train
    dW += reg * W


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
