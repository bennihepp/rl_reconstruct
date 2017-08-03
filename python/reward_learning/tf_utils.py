from __future__ import print_function

import threading
import Queue
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers


def cpu_device_name():
    return "/cpu:0"


def gpu_device_name(index=0):
    return "/gpu:{:d}".format(index)


def get_activation_function_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    elif name == "elu":
        return tf.nn.elu
    elif name == "tanh":
        return tf.nn.tanh
    elif name == "sigmoid":
        return tf.nn.sigmoid
    elif name == "identity":
        return tf.identity
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Unknown activation function: {}".format(name))


def get_optimizer_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "adam":
        return tf.train.AdamOptimizer
    elif name == "sgd":
        return tf.train.GradientDescentOptimizer
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer
    elif name == "adadelta":
        return tf.train.AdadeltaOptimizer
    elif name == "momentum":
        return tf.train.MomentumOptimizer
    elif name == "adagrad":
        return tf.train.AdagradOptimizer
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(name))


def get_weights_initializer_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "relu_uniform":
        return relu_uniform_weights_initializer
    elif name == "relu_normal":
        return relu_normal_weights_initializer
    elif name == "xavier_uniform":
        return xavier_uniform_weights_initializer
    elif name == "xavier_normal":
        return xavier_normal_weights_initializer
    else:
        raise NotImplementedError("Unknown weights initializer: {}".format(name))


def variable_size(var):
    return np.prod(var.get_shape().as_list())


def flatten_batch(x):
    """Fuse all but the first dimension, assuming that we're dealing with a batch"""
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def reshape_batch_3d_to_2d(x):
    """Fuse the last two 3d-dimensions to a single dimension.
    I.e. if input has shape [N, X, Y, Z, C] the output has shape [N, X, Y * Z, C]."""
    return tf.reshape(x, [-1, int(x.get_shape()[1]), np.prod(x.get_shape().as_list()[2:-1]), int(x.get_shape()[-1])])


def batch_norm_on_flat(x, *args, **kwargs):
    prev_x_shape = x.shape
    x = tf_layers.batch_norm(flatten_batch(x), *args, **kwargs)
    x = tf.reshape(x, prev_x_shape)
    return x


def batch_norm_on_3d(x, *args, **kwargs):
    prev_x_shape = x.shape
    x = tf_layers.batch_norm(reshape_batch_3d_to_2d(x), *args, **kwargs)
    x = tf.reshape(x, prev_x_shape)
    return x


def xavier_uniform_weights_initializer():
    # According to
    # Xavier Glorot and Yoshua Bengio (2010):
    # Understanding the difficulty of training deep feedforward neural networks.
    # International conference on artificial intelligence and statistics.
    # w_bound = np.sqrt(6. / (fan_in + fan_out))
    # return tf.random_uniform_initializer(-w_bound, w_bound)
    return tf_layers.xavier_initializer(uniform=True)


def xavier_normal_weights_initializer():
    # According to
    # Xavier Glorot and Yoshua Bengio (2010):
    # Understanding the difficulty of training deep feedforward neural networks.
    # International conference on artificial intelligence and statistics.
    # w_bound = np.sqrt(2. / (fan_in + fan_out))
    # return tf.random_normal_initializer(-w_bound, w_bound)
    return tf_layers.xavier_initializer(uniform=False)


def relu_uniform_weights_initializer():
    # According to https://arxiv.org/pdf/1502.01852.pdf (arXiv:1502.01852)
    # with modification
    # w_bound = np.sqrt(6. / fan_in)
    # return tf.random_uniform_initializer(-w_bound, w_bound)
    return tf_layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=True)


def relu_normal_weights_initializer():
    # According to https://arxiv.org/pdf/1502.01852.pdf (arXiv:1502.01852)
    # with modification
    # w_bound = np.sqrt(2. / fan_in)
    # return tf.random_normal_initializer(-w_bound, w_bound)
    return tf_layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False)


def conv3d(x, num_filters, filter_size=(3, 3, 3), stride=(1, 1, 1),
           activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
           batch_norm_after_activation=False, is_training=True,
           weights_initializer=None, padding="SAME", dtype=tf.float32, name=None, collections=None):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]

        if weights_initializer is None:
            if activation_fn == tf.nn.relu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = tf.get_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections)
        print("{}: {}".format(w.name, w.shape))
        conv_out = tf.nn.conv3d(x, w, strides, padding, name="conv3d_out")

        if add_bias:
            b = tf.get_variable("biases", [1, 1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))
            conv_out = tf.add(conv_out, b, name="linear_out")
        else:
            # Generate tensor with zero elements (so we can retrieve the biases anyway)
            b = tf.get_variable("biases", [1, 1, 1, 1, 0], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        fused=True,
                                        scope="bn",
                                        variables_collections=collections)

        if activation_fn is not None:
            conv_out = activation_fn(conv_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        fused=True,
                                        scope="bn",
                                        variables_collections=collections)

        return conv_out


def fully_connected(x, num_units, activation_fn=tf.nn.relu,
                    add_bias=False, use_batch_norm=True, batch_norm_after_activation=False, is_training=True,
                    weights_initializer=None, dtype=tf.float32, name=None, collections=None):
    if use_batch_norm:
        assert(not add_bias)

    if name is None:
        name = tf.get_variable_scope()
    with tf.variable_scope(name):
        x = flatten_batch(x)

        if weights_initializer is None:
            if activation_fn == tf.nn.relu:
                weights_initializer = relu_normal_weights_initializer()
            else:
                weights_initializer = xavier_uniform_weights_initializer()

        w = tf.get_variable("weights", [x.shape[-1], num_units], dtype, initializer=weights_initializer,
                            collections=collections)
        print("{}: {}".format(w.name, w.shape))
        out = tf.matmul(x, w, name="linear_out")

        if add_bias:
            b = tf.get_variable("biases", [num_units], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))
            out = tf.add(out, b, name="output")
        else:
            # Generate tensor with zero elements (so we can retrieve the biases anyway)
            b = tf.get_variable("biases", [0], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     fused=True,
                                     scope="bn",
                                     variables_collections=collections)

        if activation_fn is not None:
            out = activation_fn(out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     fused=True,
                                     scope="bn",
                                     variables_collections=collections)

        return out