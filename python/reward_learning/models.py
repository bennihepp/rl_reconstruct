from __future__ import print_function

import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tf_utils


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
                 add_bias=False,
                 use_batch_norm=False,
                 dropout_rate=0.2,
                 is_training=True,
                 create_summaries=None,
                 variables_collections=None):
        if use_batch_norm:
            assert not add_bias

        if not maxpool_after_each_block:
            assert(max_num_blocks > 0)

        if max_num_blocks <= 0:
            assert(max_output_grid_size > 0)

        scope = tf.get_variable_scope()

        if create_summaries is None:
            create_summaries = not scope.reuse

        self.input = input

        self.layers = []
        self.summaries = []
        self.summary_dict = {}

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
                x = tf_utils.conv3d(x, num_filters, filter_size, stride,
                                    activation_fn=activation_fn, add_bias=add_bias,
                                    use_batch_norm=False,
                                    name="conv3d_{}_{}".format(i, j),
                                    collections=variables_collections)
                activations_list.append(x)
                if use_batch_norm:
                    prev_x_shape = x.shape
                    x = tf_layers.batch_norm(tf_utils.reshape_batch_3d_to_2d(x),
                                             center=True, scale=True,
                                             is_training=is_training,
                                             fused=True,
                                             scope="conv3d_{}_{}/bn".format(i, j),
                                             variables_collections=variables_collections)
                    x = tf.reshape(x, prev_x_shape)
                # # Perform activation after batch norm
                # x = activation_fn(x, name="activation")
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
                    if use_batch_norm:
                        beta = tf.get_variable("conv3d_{}_{}/bn/beta".format(i, j))
                        biases = tf.identity(beta, name="conv3d_{}_{}/biases".format(i, j))
                    else:
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
                        layer_summary_dict = {}
                        for j in xrange(num_convs_per_block):
                            layer_summaries += [
                                tf.summary.histogram("conv3d_{}_{}/weights".format(i, j), weights_list[j]),
                                tf.summary.histogram("conv3d_{}_{}/biases".format(i, j), biases_list[j]),
                                tf.summary.histogram("conv3d_{}_{}/activations".format(i, j), activations_list[j]),
                            ]
                            layer_summary_dict.update({
                                weights_list[j].name: weights_list[j],
                                biases_list[j].name: biases_list[j],
                                activations_list[j].name: activations_list[j],
                            })
                    self.summaries.extend(layer_summaries)
                    self.summary_dict.update(layer_summary_dict)

            num_filters += filter_increase_per_block

            if max_num_blocks > 0 and (i + 1) >= max_num_blocks:
                done = True
            i += 1

        self.output_wo_dropout = x

        if dropout_rate > 0:
            keep_prob = 1 - dropout_rate
            x = tf.nn.dropout(x, keep_prob=keep_prob, name="dropout")
            print("x.shape, dropout:", x.shape)

        if create_summaries:
            with tf.device("/cpu:0"):
                dropout_summary = tf.summary.histogram("dropout_activations", x)
            self.summaries.append(dropout_summary)
            self.summary_dict.update({x.name: x})

        self.output = x


class RegressionOutputLayer(object):

    def __init__(self,
                 input,
                 num_outputs,
                 num_units=[1024],
                 activation_fn=tf.nn.relu,
                 use_batch_norm=False,
                 is_training=True,
                 create_summaries=None,
                 variables_collections=None):
        scope = tf.get_variable_scope()

        if create_summaries is None:
            create_summaries = not scope.reuse

        self.input = input

        self.layers = []
        self.summaries = []
        self.summary_dict = {}

        x = self.input
        # x = tf_utils.flatten(x)
        # print("x.shape, flat:", x.shape)

        for i in xrange(len(num_units)):
            with tf.variable_scope('fc_{}'.format(i)) as scope2:
                # if use_batch_norm:
                #     bias_initializer = None
                # else:
                #     bias_initializer = tf.zeros_initializer()
                # activations = tf_layers.fully_connected(x, num_units[i],
                #                                         activation_fn=activation_fn,
                #                                         weights_initializer=tf_layers.xavier_initializer(),
                #                                         biases_initializer=bias_initializer)
                add_bias = not use_batch_norm
                activations = tf_utils.fully_connected(x, num_units[i],
                                                       activation_fn=activation_fn, add_bias=add_bias,
                                                       use_batch_norm=False,
                                                       collections=variables_collections)
                x = activations
                if use_batch_norm:
                    x = tf_utils.flatten_batch(x)
                    x = tf_layers.batch_norm(x,
                                             center=True, scale=True,
                                             is_training=is_training,
                                             fused=True,
                                             scope="bn",
                                             variables_collections=variables_collections)
                # Perform activation after batch norm
                # x = activation_fn(x, name="activation")
                print("activations.shape, fc_{}:".format(i), activations.shape)
            with tf.variable_scope(scope2, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                print("{}: {}".format(weights.name, weights.shape))
                if use_batch_norm:
                    beta = tf.get_variable("bn/beta")
                    biases = tf.identity(beta, name="biases")
                else:
                    biases = tf.get_variable('biases')
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
                        layer_summary_dict = {
                            weights.name: weights,
                            biases.name: biases,
                            activations.name: activations,
                        }
                    self.summaries.extend(layer_summaries)
                    self.summary_dict.update(layer_summary_dict)

        with tf.variable_scope('output') as scope2:
            # activations = tf_layers.fully_connected(x, num_outputs,
            #                                    activation_fn=None,
            #                                    weights_initializer=tf_layers.xavier_initializer(),
            #                                    biases_initializer=tf.zeros_initializer())
            add_bias = True
            x = tf_utils.fully_connected(x, num_outputs,
                                         activation_fn=None, add_bias=add_bias,
                                         use_batch_norm=False,
                                         collections=variables_collections)
            activations = x
            print("activations.shape, fc_{}:".format(i), activations.shape)
            print("output.shape:", activations.shape)
            with tf.variable_scope(scope2, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                print("{}: {}".format(weights.name, weights.shape))
                biases = tf.get_variable('biases')
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
                        layer_summary_dict = {
                            weights.name: weights,
                            biases.name: biases,
                            activations.name: activations,
                        }
                    self.summaries.extend(layer_summaries)
                    self.summary_dict.update(layer_summary_dict)

        self.output = x


class ModelModule(object):

    def __init__(self, config):
        self._config = config
        self._variables = []
        self._summaries = {}
        self._scope = tf.get_variable_scope()
        self._output = None

    def _get_config(self, name, default=None):
        if isinstance(self._config, dict):
            return self._config[name]
        else:
            return getattr(self._config, name, default)

    def _get_config_list(self, name, converter, default=None):
        config_value = self._get_config(name, default)
        if config_value is None:
            return default
        config_list = [converter(x) for x in config_value.strip("[]").split(',')]
        return config_list

    def _get_config_int_list(self, name, default=None):
        return self._get_config_list(name, int, default)

    def _get_config_float_list(self, name, default=None):
        return self._get_config_list(name, float, default)

    def _get_activation_fn(self, fn, default=tf.nn.relu):
        if isinstance(fn, types.FunctionType):
            return fn
        else:
            return tf_utils.get_activation_function_by_name(fn, default)

    def _add_variable(self, var):
        self._variables.append(var)

    def _add_variables(self, vars):
        self._variables.extend(vars)

    def _add_summary(self, name, tensor):
        assert(name not in self._summaries)
        self._summaries[name] = tensor

    def _set_output(self, output):
        self._output = output

    @property
    def variables(self):
        return self._variables

    @property
    def summaries(self):
        return self._summaries

    @property
    def scope(self):
        return self._scope

    @property
    def output(self):
        return self._output


class Conv3DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_collections=None):
        super(Conv3DModule, self).__init__(config)

        num_convs_per_block = self._get_config("num_convs_per_block", 2)
        maxpool_after_each_block = self._get_config("maxpool_after_each_block", True)
        initial_num_filters = self._get_config("initial_num_filters", 8)
        filter_increase_per_block = self._get_config("filter_increase_per_block", 8)
        filter_increase_within_block = self._get_config("filter_increase_within_block", 0)
        max_num_blocks = self._get_config("max_num_blocks", -1)
        max_output_grid_size = self._get_config("max_output_grid_size", 8)
        dropout_rate = self._get_config("dropout_rate", 0.3)
        add_bias = self._get_config("add_bias_3dconv")
        use_batch_norm = self._get_config("use_batch_norm_conv3d", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn_conv3d"))

        if use_batch_norm:
            assert not add_bias

        if not maxpool_after_each_block:
            assert(max_num_blocks > 0)

        if max_num_blocks <= 0:
            assert(max_output_grid_size > 0)

        x = input
        print("input:", x.shape)

        num_filters = initial_num_filters
        i = 0
        done = False
        while not done:
            filter_size = [3, 3, 3]
            stride = [1, 1, 1]
            for j in xrange(num_convs_per_block):
                x = tf_utils.conv3d(x, num_filters, filter_size, stride,
                                    activation_fn=activation_fn, add_bias=add_bias,
                                    use_batch_norm=use_batch_norm,
                                    is_training=is_training,
                                    name="conv3d_{}_{}".format(i, j),
                                    collections=variables_collections)
                self._add_summary("conv3d_{}_{}/activations".format(i, j), x)
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
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in xrange(num_convs_per_block):
                    weights = tf.get_variable("conv3d_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv3d_{}_{}/bn/beta".format(i, j))
                    else:
                        biases = tf.get_variable("conv3d_{}_{}/biases".format(i, j))
                    # Add summaries for layer
                    self._add_summary("conv3d_{}_{}/weights".format(i, j), weights)
                    self._add_summary("conv3d_{}_{}/biases".format(i, j), biases)

                self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

            num_filters += filter_increase_per_block

            if max_num_blocks > 0 and (i + 1) >= max_num_blocks:
                done = True
            i += 1

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            x = tf.nn.dropout(x, keep_prob=keep_prob, name="dropout")
            print("x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class RegressionModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 num_outputs,
                 is_training=True,
                 variables_collections=None):
        super(RegressionModule, self).__init__(config)

        num_units = self._get_config_int_list("num_units_regression")
        assert(num_units is not None)
        use_batch_norm = self._get_config("use_batch_norm_regression", True)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        use_fully_convolutional = self._get_config("use_fully_convolutional", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn_regression"))

        x = input

        for i in xrange(len(num_units)):
            with tf.variable_scope('fc_{}'.format(i)) as scope:
                if use_fully_convolutional:
                    filter_size = [int(s) for s in x.shape[1:-1]]
                    stride = [1, 1, 1]
                    x = tf_utils.conv3d(
                        x, num_units[i], filter_size, stride,
                        activation_fn=activation_fn,
                        add_bias=add_bias,
                        use_batch_norm=use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        padding="VALID",
                        collections=variables_collections)
                else:
                    x = tf_utils.fully_connected(
                        x, num_units[i],
                        activation_fn=activation_fn,
                        add_bias=add_bias,
                        use_batch_norm=use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        collections=variables_collections)
                print("activations.shape, fc_{}:".format(i), x.shape)
                self._add_summary("fc_{}/activations".format(i), x)
            with tf.variable_scope(scope, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                print("{}: {}".format(weights.name, weights.shape))
                if use_batch_norm:
                    biases = tf.get_variable("bn/beta")
                else:
                    biases = tf.get_variable('biases')
                print("{}: {}".format(biases.name, biases.shape))
                # Add summaries for layer
                self._add_summary("fc_{}/weights".format(i), weights)
                self._add_summary("fc_{}/biases".format(i), biases)
            self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

        with tf.variable_scope('output') as scope:
            if use_fully_convolutional:
                filter_size = [int(s) for s in x.shape[1:-1]]
                stride = [1, 1, 1]
                x = tf_utils.conv3d(x, num_outputs, filter_size, stride,
                                    activation_fn=None,
                                    add_bias=True,
                                    use_batch_norm=False,
                                    is_training=is_training,
                                    padding="VALID",
                                    collections=variables_collections)
                if int(x.shape[1]) == 1 and int(x.shape[2]) == 1 and int(x.shape[3]) == 1:
                    # Remove 3D tensor dimensions if there is no spatial extent.
                    # Otherwise reductions might lead to non-expected results.
                    x = tf.reshape(x, [int(x.shape[0]), -1])
            else:
                x = tf_utils.fully_connected(x, num_outputs,
                                             activation_fn=None, add_bias=True,
                                             use_batch_norm=False,
                                             is_training=is_training,
                                             collections=variables_collections)
            self._add_summary("output/activations", x)
            print("output.shape:", x.shape)
        with tf.variable_scope(scope, reuse=True):
            # Make layer accessible from outside
            weights = tf.get_variable('weights')
            print("{}: {}".format(weights.name, weights.shape))
            biases = tf.get_variable('biases')
            print("{}: {}".format(biases.name, biases.shape))
            # Add summaries for layer
            self._add_summary("output/weights", weights)
            self._add_summary("output/biases", biases)
        self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

        self._set_output(x)


class Model(ModelModule):

    def __init__(self, config, input, target_shape, is_training=True, variables_collections=None):
        super(Model, self).__init__(config)

        assert(len(target_shape) == 1)
        num_outputs = target_shape[-1]

        self._conv3d_module = Conv3DModule(
            self._config,
            input,
            is_training=is_training,
            variables_collections=variables_collections)
        self._regression_module = RegressionModule(
            self._config,
            self._conv3d_module.output,
            num_outputs,
            is_training=is_training,
            variables_collections=variables_collections)

        self._set_output(self._regression_module.output)
        self._add_variables(self._conv3d_module.variables)
        self._add_variables(self._regression_module.variables)
        for name, tensor in self._conv3d_module.summaries.iteritems():
            self._add_summary("conv3d/" + name, tensor)
        for name, tensor in self._regression_module.summaries.iteritems():
            self._add_summary("regression/" + name, tensor)
        self._modules = {"conv3d": self._conv3d_module,
                         "regression": self._regression_module}

    @property
    def modules(self):
        return self._modules
