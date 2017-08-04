from __future__ import print_function

import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.training import moving_averages
from tensorflow.python.client import device_lib


def get_available_device_names():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def get_available_gpu_names():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_gpu_ids():
    gpu_names = get_available_gpu_names()
    gpu_id_regex = re.compile("/gpu:(\d)")
    gpu_ids = []
    for gpu_name in gpu_names:
        match = gpu_id_regex.match(gpu_name)
        gpu_id = int(match.group(1))
        gpu_ids.append(gpu_id)
    return gpu_ids


def cpu_device_name():
    return "/cpu:0"


def gpu_device_name(index=0):
    return "/gpu:{:d}".format(index)


def tf_device_name(device_id=0):
    if device_id >= 0:
        return gpu_device_name(device_id)
    elif device_id == -1:
        return cpu_device_name()
    else:
        raise NotImplementedError("Unknown device id: {}".format(device_id))


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


def batch_norm_custom(
        inputs,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        is_training=True,
        reuse=None,
        variables_collections=None,
        trainable=True,
        zero_debias_moving_mean=False,
        variables_on_cpu=False,
        scope="bn"):
    if scope is None:
        scope = tf.get_variable_scope()

    original_shape = inputs.shape
    inputs = reshape_batch_3d_to_2d(inputs)
    params_shape = [inputs.shape[-1]]
    dtype = inputs.dtype

    with tf.variable_scope(scope, reuse=reuse):

        # Allocate parameters for the beta and gamma of the normalization.
        trainable_beta = trainable and center
        beta = get_tf_variable("beta", shape=params_shape, dtype=dtype,
                               initializer=tf.zeros_initializer(), trainable=trainable_beta,
                               collections=variables_collections, variable_on_cpu=variables_on_cpu)

        trainable_gamma = trainable and scale
        gamma = get_tf_variable("gamma", shape=params_shape, dtype=dtype,
                                initializer=tf.ones_initializer(), trainable=trainable_gamma,
                                collections=variables_collections, variable_on_cpu=variables_on_cpu)

        # Moving mean and variance
        moving_mean = get_tf_variable(
            "moving_mean",
            shape=params_shape,
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=variables_collections,
            variable_on_cpu=variables_on_cpu)
        moving_variance = get_tf_variable(
            "moving_variance",
            shape=params_shape,
            dtype=dtype,
            initializer=tf.ones_initializer(),
            trainable=False,
            collections=variables_collections,
            variable_on_cpu=variables_on_cpu)

    def _fused_batch_norm_training():
        return tf.nn.fused_batch_norm(
            inputs, gamma, beta, epsilon=epsilon)

    def _fused_batch_norm_inference():
        return tf.nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            mean=moving_mean,
            variance=moving_variance,
            epsilon=epsilon,
            is_training=False)

    if is_training:
        outputs, mean, variance = _fused_batch_norm_training()
    else:
        outputs, mean, variance = _fused_batch_norm_inference()

    need_updates = is_training
    if need_updates:
        if updates_collections is None:
            def _force_updates():
                """Internal function forces updates moving_vars if is_training."""
                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(outputs)
            outputs = _force_updates()
        else:
            def _delay_updates():
                """Internal function that delay updates moving_vars if is_training."""
                update_moving_mean = moving_averages.assign_moving_average(
                    moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)
                return update_moving_mean, update_moving_variance
            update_mean, update_variance = _delay_updates()
            tf.add_to_collection(updates_collections, update_mean)
            tf.add_to_collection(updates_collections, update_variance)

    outputs = tf.reshape(outputs, original_shape)
    return outputs


def batch_norm_on_flat(inputs, *args, **kwargs):
    variables_on_cpu = kwargs.get("variables_on_cpu", False)
    if "variables_on_cpu" in kwargs:
        del kwargs["variables_on_cpu"]
    if variables_on_cpu:
        kwargs["variables_on_cpu"] = variables_on_cpu
        return batch_norm_custom(inputs, *args, **kwargs)
    return batch_norm_custom(inputs, *args, **kwargs)
    # Use tensorflow batch norm implementation
    # original_shape = inputs.shape
    # outputs = tf_layers.batch_norm(flatten_batch(inputs), *args, **kwargs)
    # outputs = tf.reshape(outputs, original_shape)
    # return outputs


def batch_norm_on_3d(inputs, *args, **kwargs):
    variables_on_cpu = kwargs.get("variables_on_cpu", False)
    if "variables_on_cpu" in kwargs:
        del kwargs["variables_on_cpu"]
    if variables_on_cpu:
        kwargs["variables_on_cpu"] = variables_on_cpu
        return batch_norm_custom(inputs, *args, **kwargs)
    return batch_norm_custom(inputs, *args, **kwargs)
    # Use tensorflow batch norm implementation
    # original_shape = inputs.shape
    # outputs = tf_layers.batch_norm(reshape_batch_3d_to_2d(inputs), *args, **kwargs)
    # outputs = tf.reshape(outputs, original_shape)
    # return outputs


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


def get_tf_variable(name, shape=None, dtype=None, initializer=None, collections=None, variable_on_cpu=False, **kwargs):
    if variable_on_cpu:
        with tf.device(cpu_device_name()):
            return tf.get_variable(name, shape, dtype, initializer, collections, **kwargs)
    else:
        return tf.get_variable(name, shape, dtype, initializer, collections, **kwargs)


def conv3d(x, num_filters, filter_size=(3, 3, 3), stride=(1, 1, 1),
           activation_fn=tf.nn.relu, add_bias=False, use_batch_norm=True,
           batch_norm_after_activation=False, is_training=True,
           weights_initializer=None, padding="SAME",
           dtype=tf.float32, name=None, variables_on_cpu=False, collections=None):
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

        w = get_tf_variable("weights", filter_shape, dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        print("{}: {}".format(w.name, w.shape))
        conv_out = tf.nn.conv3d(x, w, strides, padding, name="conv3d_out")

        if add_bias:
            b = get_tf_variable("biases", [1, 1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            print("{}: {}".format(b.name, b.shape))
            conv_out = tf.add(conv_out, b, name="linear_out")
        else:
            # Generate tensor with zero elements (so we can retrieve the biases anyway)
            b = get_tf_variable("biases", [1, 1, 1, 1, 0], initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections,
                                        variables_on_cpu=variables_on_cpu)

        if activation_fn is not None:
            conv_out = activation_fn(conv_out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            conv_out = batch_norm_on_3d(conv_out,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope="bn",
                                        variables_collections=collections)

        return conv_out


def fully_connected(x, num_units, activation_fn=tf.nn.relu,
                    add_bias=False, use_batch_norm=True, batch_norm_after_activation=False, is_training=True,
                    weights_initializer=None, dtype=tf.float32, name=None, variables_on_cpu=False, collections=None):
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

        w = get_tf_variable("weights", [x.shape[-1], num_units], dtype, initializer=weights_initializer,
                            collections=collections, variable_on_cpu=variables_on_cpu)
        print("{}: {}".format(w.name, w.shape))
        out = tf.matmul(x, w, name="linear_out")

        if add_bias:
            b = get_tf_variable("biases", [num_units], initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            print("{}: {}".format(b.name, b.shape))
            out = tf.add(out, b, name="output")
        else:
            # Generate tensor with zero elements (so we can retrieve the biases anyway)
            b = get_tf_variable("biases", [0], initializer=tf.zeros_initializer(),
                                collections=collections, variable_on_cpu=variables_on_cpu)
            print("{}: {}".format(b.name, b.shape))

        if use_batch_norm and not batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     scope="bn",
                                     variables_collections=collections)

        if activation_fn is not None:
            out = activation_fn(out, name="activation")

        if use_batch_norm and batch_norm_after_activation:
            out = batch_norm_on_flat(out,
                                     center=True, scale=True,
                                     is_training=is_training,
                                     scope="bn",
                                     variables_collections=collections)

        return out


class ModelHistogramSummary(object):

    def __init__(self, name_tensor_dict):
        self._fetches = []
        self._placeholders = []
        summaries = []
        for name, tensor in name_tensor_dict.iteritems():
            var_batch_shape = [None] + tensor.shape[1:].as_list()
            placeholder = tf.placeholder(tensor.dtype, var_batch_shape)
            summaries.append(tf.summary.histogram(name, placeholder))
            self._fetches.append(tensor)
            self._placeholders.append(placeholder)
        self._summary_op = tf.summary.merge(summaries)

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def fetches(self):
        return self._fetches

    @property
    def summary_op(self):
        return self._summary_op
