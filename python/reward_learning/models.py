from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv3d(x, num_filters, name, filter_size=(3, 3, 3), stride=(1, 1, 1),
           add_bias=True, padding="SAME",
           dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        strides = [1, stride[0], stride[1], stride[2], 1]
        filter_shape = [filter_size[0], filter_size[1], filter_size[2], int(x.get_shape()[4]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("weights", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        print("{}: {}".format(w.name, w.shape))
        # w = tf.get_variable("W", filter_shape, dtype, tf_layers.xavier_initializer(),
        #                     collections=collections)
        conv_out = tf.nn.conv3d(x, w, strides, padding)

        if add_bias:
            b = tf.get_variable("biases", [1, 1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))
            return conv_out + b
        else:
            # Generate tensor with zero elements (so we can retrieve the biases anyway)
            b = tf.get_variable("biases", [1, 1, 1, 1, 0], initializer=tf.zeros_initializer(),
                                collections=collections)
            print("{}: {}".format(b.name, b.shape))
            return conv_out


class Conv3DLayers(object):

    def __init__(self,
                 input,
                 num_convs_per_block=2,
                 initial_num_filters=8,
                 filter_increase_per_block=8,
                 filter_increase_within_block=0,
                 maxpool_after_each_block=False,
                 max_num_blocks=-1,
                 max_output_grid_size=8,
                 activation_fn=tf.nn.relu,
                 add_biases=False,
                 dropout_rate=0.2,
                 reuse=False,
                 create_summaries=None):
        if create_summaries is None:
            create_summaries = not reuse

        if not maxpool_after_each_block:
            assert(max_num_blocks > 0)

        if max_num_blocks <= 0:
            assert(max_output_grid_size > 0)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
            self.input = input

            self.layers = []
            self.summaries = []

            x = self.input
            print("input:", x.shape)

            num_filters = initial_num_filters
            i = 0
            done = False
            while not done:
                filter_size = [3, 3, 3]
                stride = [1, 1, 1]
                activations_list = []
                for j in xrange(num_convs_per_block):
                    x = activation_fn(conv3d(x, num_filters, "conv3d_{}_{}".format(i, j),
                                             filter_size, stride, add_bias=add_biases))
                    activations_list.append(x)
                    if (j + 1) < num_convs_per_block:
                        num_filters += filter_increase_within_block
                do_maxpool = maxpool_after_each_block or (max_num_blocks > 0 and (i + 1) >= max_num_blocks)
                if do_maxpool:
                    x_grid_size = int(x.shape[-2])
                    if x_grid_size > max_output_grid_size:
                        ksize = [1, 2, 2, 2, 1]
                        strides = [1, 2, 2, 2, 1]
                        with tf.variable_scope("maxpool3d_{}".format(i)):
                            x = tf.nn.max_pool3d(x, ksize, strides, padding="SAME")
                    elif max_output_grid_size > 0:
                        done = True
                print("x.shape:", i, x.shape)
                # Make layer accessible from outside
                with tf.variable_scope(scope, reuse=True) as scope2:
                    weights_list = []
                    biases_list = []
                    for j in xrange(num_convs_per_block):
                        weights = tf.get_variable("conv3d_{}_{}/weights".format(i, j))
                        biases = tf.get_variable("conv3d_{}_{}/biases".format(i, j))
                        weights_list.append(weights)
                        biases_list.append(biases)
                        layer = (weights, biases, activations_list[j])
                        self.layers.append(layer)
                with tf.variable_scope(scope) as scope2:
                    if create_summaries:
                        # Create summaries for layer
                        with tf.device("/cpu:0"):
                            layer_summaries = []
                            for j in xrange(num_convs_per_block):
                                layer_summaries += [
                                    tf.summary.histogram("conv3d_{}_{}/weights".format(i, j), weights_list[j]),
                                    tf.summary.histogram("conv3d_{}_{}/biases".format(i, j), biases_list[j]),
                                    tf.summary.histogram("conv3d_{}_{}/activations".format(i, j), activations_list[j]),
                                ]
                        self.summaries.extend(layer_summaries)

                num_filters += filter_increase_per_block

                if max_num_blocks > 0 and (i + 1) >= max_num_blocks:
                    done = True
                i += 1

            self.output_wo_dropout = x

            if dropout_rate > 0:
                with tf.variable_scope("dropout") as scope2:
                    keep_prob = 1 - dropout_rate
                    x = tf.nn.dropout(x, keep_prob=keep_prob)
                print("x.shape, dropout:", x.shape)

            if create_summaries:
                with tf.device("/cpu:0"):
                    dropout_summary = tf.summary.histogram("dropout_activations", x)
                self.summaries.append(dropout_summary)

            self.output = x


class RegressionOutputLayer(object):

    def __init__(self,
                 input,
                 num_outputs,
                 num_units=[1024],
                 activation_fn=tf.nn.relu,
                 reuse=False,
                 create_summaries=None):
        if create_summaries is None:
            create_summaries = not reuse

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
            self.input = input

            self.layers = []
            self.summaries = []

            x = self.input
            x = flatten(x)
            print("x.shape, flat:", x.shape)

            for i in xrange(len(num_units)):
                with tf.variable_scope('fc_{}'.format(i)) as scope2:
                    activations = tf_layers.fully_connected(x, num_units[i],
                                                            activation_fn=activation_fn,
                                                            weights_initializer=tf_layers.xavier_initializer(),
                                                            biases_initializer=tf.zeros_initializer())
                    print("activations.shape, fc_{}:".format(i), activations.shape)
                    x = activations
                with tf.variable_scope(scope2, reuse=True):
                    # Make layer accessible from outside
                    weights = tf.get_variable('fully_connected/weights'.format(i))
                    print("{}: {}".format(weights.name, weights.shape))
                    biases = tf.get_variable('fully_connected/biases'.format(i))
                    print("{}: {}".format(biases.name, biases.shape))
                    layer = (weights, biases, activations)
                    self.layers.append(layer)
                with tf.variable_scope(scope2):
                    if create_summaries:
                        # Create summaries for layer
                        with tf.device("/cpu:0"):
                            layer_summaries = [
                                tf.summary.histogram("weights".format(i), weights),
                                tf.summary.histogram("biases".format(i), biases),
                                tf.summary.histogram("activations".format(i), activations),
                            ]
                        self.summaries.extend(layer_summaries)

            with tf.variable_scope('output') as scope2:
                activations = tf_layers.fully_connected(x, num_outputs,
                                                   activation_fn=None,
                                                   weights_initializer=tf_layers.xavier_initializer(),
                                                   biases_initializer=tf.zeros_initializer())
                x = activations
                print("output.shape:", activations.shape)
                with tf.variable_scope(scope2, reuse=True):
                    # Make layer accessible from outside
                    weights = tf.get_variable('fully_connected/weights')
                    print("{}: {}".format(weights.name, weights.shape))
                    biases = tf.get_variable('fully_connected/biases')
                    print("{}: {}".format(biases.name, biases.shape))
                    layer = (weights, biases, activations)
                    self.layers.append(layer)
                with tf.variable_scope(scope2):
                    if create_summaries:
                        # Create summaries for layer
                        with tf.device("/cpu:0"):
                            layer_summaries = [
                                tf.summary.histogram("weights", weights),
                                tf.summary.histogram("biases", biases),
                                tf.summary.histogram("activations", activations),
                            ]
                        self.summaries.extend(layer_summaries)

            self.output = x
