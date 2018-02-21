#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/Users/jiahonghe/Desktop/Deep\ Learning/Deep\ Learning-Spring-2018/assignment2/ecbm4040/neuralnets')
from layers import *

####################################
# TODO: Build your own LeNet model #
####################################

def my_LeNet(input_x, input_y,
             img_len=32, channel_num=3, output_size=10,
             conv_featmap=[6, 16], fc_units=[84],
             conv_kernel_size=[5, 5], pooling_size=[2, 2],
             l2_norm=0.01, seed=235
             ):

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    # flatten
    pool_shape = pooling_layer_0.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_0.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)

    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_1.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_1.output(), loss
    raise NotImplementedError

####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################


def my_training(X_train, y_train, X_val, y_val):

    raise NotImplementedError
##########################################
#            End of your code            #
##########################################


def my_training_task4():
    # TODO: Copy my_training function, make modifications so that it uses your
    # data generator from task 4 to train.
    raise NotImplementedError
