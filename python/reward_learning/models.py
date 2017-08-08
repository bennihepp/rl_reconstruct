from __future__ import print_function

import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tf_utils


class ModelModule(object):

    def __init__(self, config, input_tensor):
        self._config = config
        self._input = input_tensor
        self._variables = []
        self._summaries = {}
        self._scope = tf.get_variable_scope()
        self._output = None

    def _get_config(self, name, default=None):
        if isinstance(self._config, dict):
            return self._config.get(name, default)
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

    def _add_summaries(self, summary_dict):
        for name, tensor in summary_dict.iteritems():
            assert(name not in self._summaries)
            self._summaries[name] = tensor

    def _set_output(self, output):
        self._output = output

    @property
    def config(self):
        return self._config

    @property
    def input(self):
        return self._input

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
                 variables_on_cpu=False,
                 variables_collections=None,
                 verbose=False):
        super(Conv3DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 2)
        maxpool_after_each_block = self._get_config("maxpool_after_each_block", True)
        initial_num_filters = self._get_config("initial_num_filters", 8)
        filter_increase_per_block = self._get_config("filter_increase_per_block", 8)
        filter_increase_within_block = self._get_config("filter_increase_within_block", 0)
        max_num_blocks = self._get_config("max_num_blocks", -1)
        max_output_grid_size = self._get_config("max_output_grid_size", 8)
        dropout_rate = self._get_config("dropout_rate", 0.3)
        add_bias = self._get_config("add_bias")
        use_batch_norm = self._get_config("use_batch_norm", True)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        if use_batch_norm:
            assert not add_bias

        if not maxpool_after_each_block:
            assert(max_num_blocks > 0)

        if max_num_blocks <= 0:
            assert(max_output_grid_size > 0)

        x = input
        if verbose:
            print("--- Conv3DModule ---")
            print("  input:", x.shape)

        num_filters = initial_num_filters
        i = 0
        done = False
        while not done:
            filter_size = [3, 3, 3]
            stride = [1, 1, 1]
            for j in xrange(num_convs_per_block):
                x = tf_utils.conv3d(
                    x, num_filters, filter_size, stride,
                    activation_fn=activation_fn,
                    add_bias=add_bias,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="conv3d_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
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
            if verbose:
                print("  x.shape:", i, x.shape)
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
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class RegressionModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 num_outputs,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 verbose=False):
        super(RegressionModule, self).__init__(config, input)

        num_units = self._get_config_int_list("num_units")
        assert(num_units is not None)
        use_batch_norm = self._get_config("use_batch_norm", True)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        fully_convolutional = self._get_config("fully_convolutional", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        x = input
        if verbose:
            print("--- RegressionModule ---")
            print("  input:", x.shape)

        for i in xrange(len(num_units)):
            with tf.variable_scope('fc_{}'.format(i)) as scope:
                if fully_convolutional:
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
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections)
                else:
                    x = tf_utils.fully_connected(
                        x, num_units[i],
                        activation_fn=activation_fn,
                        add_bias=add_bias,
                        use_batch_norm=use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections)
                if verbose:
                    print("  activations.shape, fc_{}:".format(i), x.shape)
                self._add_summary("fc_{}/activations".format(i), x)
            with tf.variable_scope(scope, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                if verbose:
                    print("  {}: {}".format(weights.name, weights.shape))
                if use_batch_norm:
                    biases = tf.get_variable("bn/beta")
                else:
                    biases = tf.get_variable('biases')
                if verbose:
                        print("  {}: {}".format(biases.name, biases.shape))
                # Add summaries for layer
                self._add_summary("fc_{}/weights".format(i), weights)
                self._add_summary("fc_{}/biases".format(i), biases)
            self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

        with tf.variable_scope('output') as scope:
            if fully_convolutional:
                filter_size = [int(s) for s in x.shape[1:-1]]
                stride = [1, 1, 1]
                x = tf_utils.conv3d(x, num_outputs, filter_size, stride,
                                    activation_fn=None,
                                    add_bias=True,
                                    use_batch_norm=False,
                                    is_training=is_training,
                                    padding="VALID",
                                    variables_on_cpu=variables_on_cpu,
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
                                             variables_on_cpu=variables_on_cpu,
                                             collections=variables_collections)
            self._add_summary("output/activations", x)
            if verbose:
                print("  output.shape:", x.shape)
        with tf.variable_scope(scope, reuse=True):
            # Make layer accessible from outside
            weights = tf.get_variable('weights')
            if verbose:
                print("  {}: {}".format(weights.name, weights.shape))
            biases = tf.get_variable('biases')
            if verbose:
                print("  {}: {}".format(biases.name, biases.shape))
            # Add summaries for layer
            self._add_summary("output/weights", weights)
            self._add_summary("output/biases", biases)
        self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

        self._set_output(x)


class UpsamplingModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 target_shape,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 verbose=False):
        super(UpsamplingModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 1)
        initial_num_filters = self._get_config("initial_num_filters", -1)
        filter_decrease_per_block = self._get_config("filter_decrease_per_block", 8)
        filter_decrease_within_block = self._get_config("filter_decrease_within_block", 0)
        use_batch_norm = self._get_config("use_batch_norm", False)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        activation_fn = self._get_activation_fn(self._get_config("activation_fn", None))

        if initial_num_filters <= 0:
            initial_num_filters = int(input.shape[-1])

        x = input
        if verbose:
            print("--- UpsamplingModule ---")
            print("  input:", x.shape)

        i = 0
        num_filters = initial_num_filters
        last_conv_transpose = False
        while True:
            x_grid_size = int(x.shape[-2])
            if x_grid_size >= target_shape[-2]:
                break
            filter_size = [3, 3, 3]
            for j in xrange(num_convs_per_block):
                if (j + 1) < num_convs_per_block:
                    num_filters -= filter_decrease_within_block
                    output_shape = [int(s) for s in x.shape]
                    output_shape[-1] = num_filters
                    stride = [1, 1, 1]
                else:
                    output_shape = None
                    new_output_x_size = int(x.shape[2]) * 2
                    stride = [2, 2, 2]
                    print("Last upsampling layer")
                    if (j + 1) >= num_convs_per_block and new_output_x_size >= target_shape[-2]:
                        last_conv_transpose = True
                        num_filters = target_shape[-1]
                x = tf_utils.conv3d_transpose(
                    x,
                    num_filters,
                    output_shape=output_shape,
                    filter_size=filter_size,
                    stride=stride,
                    activation_fn=activation_fn,
                    add_bias=add_bias,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="conv3d_trans_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections)
                if last_conv_transpose:
                    for k in xrange(num_filters):
                        self._add_summary("output_{}".format(k), x[..., slice(k, k+1)])
                    self._add_summary("output_all", x)
                else:
                    self._add_summary("conv3d_trans_{}_{}/activations".format(i, j), x)
                if verbose:
                    print("  x.shape, {} {}: {}".format(i, j, x.shape))
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in xrange(num_convs_per_block):
                    weights = tf.get_variable("conv3d_trans_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv3d_trans_{}_{}/bn/beta".format(i, j))
                    else:
                        biases = tf.get_variable("conv3d_trans_{}_{}/biases".format(i, j))
                    # Add summaries for layer
                    self._add_summary("conv3d_trans_{}_{}/weights".format(i, j), weights)
                    self._add_summary("conv3d_trans_{}_{}/biases".format(i, j), biases)

                self._add_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name))

            num_filters -= filter_decrease_per_block
            i += 1

        self._set_output(x)


class Model(ModelModule):

    def __init__(self, config, input_batch, target_batch, is_training=True,
                 variables_on_cpu=False, variables_collections=None, verbose=False):
        super(Model, self).__init__(config, input_batch)

        target_shape = [int(s) for s in target_batch.shape[1:]]
        num_outputs = target_shape[-1]

        self._modules = {}

        self._conv3d_module = Conv3DModule(
            self._config["conv3d"],
            input_batch,
            is_training=is_training,
            variables_on_cpu=variables_on_cpu,
            variables_collections=variables_collections,
            verbose=verbose)
        self._modules["conv3d"] = self._conv3d_module

        assert(("regression" in self._config) ^ ("upsampling" in self._config))

        if "regression" in self._config:
            self._regression_module = RegressionModule(
                self._config["regression"],
                self._conv3d_module.output,
                num_outputs,
                is_training=is_training,
                variables_on_cpu=variables_on_cpu,
                variables_collections=variables_collections,
                verbose=verbose)
            self._modules["regression"] = self._regression_module
            self._set_output(self._regression_module.output)
            self._loss_batch = tf.reduce_mean(
                tf.square(self.output - target_batch),
                axis=-1, name="loss_batch")
        else:
            self._upsampling_module = UpsamplingModule(
                self._config["upsampling"],
                self._conv3d_module.output,
                target_shape,
                is_training=is_training,
                variables_on_cpu=variables_on_cpu,
                variables_collections=variables_collections,
                verbose=verbose)
            self._modules["upsampling"] = self._upsampling_module
            self._set_output(self._upsampling_module.output)
            # Reshape output and target to compute loss
            if verbose:
                print("Shape of output: {}".format(self.output.shape))
                print("Shape of target_batch: {}".format(target_batch.shape))
            output = tf.reshape(self.output, [int(self.output.shape[0]), -1])
            reshaped_target_batch = tf.reshape(target_batch, [int(self.output.shape[0]), -1])
            if verbose:
                print("Reshaped output: {}".format(output.shape))
                print("Reshaped target_batch: {}".format(reshaped_target_batch.shape))
            self._loss_batch = tf.reduce_mean(
                tf.square(output - reshaped_target_batch),
                axis=-1, name="loss_batch")
            if verbose:
                print("Shape of batch loss: {}".format(self._loss_batch.shape))

        for module_name, module in self._modules.iteritems():
            self._add_variables(module.variables)
            for tensor_name, tensor in module.summaries.iteritems():
                self._add_summary(module_name + "/" + tensor_name, tensor)

        self._loss = tf.reduce_mean(self._loss_batch, name="loss")
        self._loss = tf.reduce_mean(tf.square(self.output - target_batch), name="loss")
        if verbose:
            print("Shape of loss: {}".format(self._loss.shape))
        self._loss_min = tf.reduce_min(self._loss_batch, name="loss_min")
        self._loss_max = tf.reduce_max(self._loss_batch, name="loss_max")

        self._gradients = tf.gradients(self._loss, self.variables)

    @property
    def gradients(self):
        return self._gradients

    @property
    def modules(self):
        return self._modules

    @property
    def loss_batch(self):
        return self._loss_batch

    @property
    def loss(self):
        return self._loss

    @property
    def loss_min(self):
        return self._loss_min

    @property
    def loss_max(self):
        return self._loss_max


class MultiGpuModelWrapper(ModelModule):

    def __init__(self, models, verbose=False):
        assert(len(models) > 0)
        super(MultiGpuModelWrapper, self).__init__(models[0].config, models[0].input)

        self._models = models
        self._add_variables(models[0].variables)
        for name in models[0].summaries.keys():
            concat_tensors = tf.concat([model.summaries[name] for model in models], axis=0)
            self._add_summary(name, concat_tensors)

        concat_output = tf.concat([model.output for model in models], axis=0)
        self._set_output(concat_output)

        sum_loss = tf.add_n([model.loss for model in models])
        mean_loss = tf.multiply(sum_loss, 1.0 / len(models))
        self._loss = mean_loss

        loss_min_concat = tf.stack([model.loss_min for model in models], axis=0)
        self._loss_min = tf.reduce_min(loss_min_concat, axis=0)

        loss_max_concat = tf.stack([model.loss_max for model in models], axis=0)
        self._loss_max = tf.reduce_max(loss_max_concat, axis=0)

        self._gradients = self._mean_gradients([model.gradients for model in models], verbose)

    def _mean_gradients(self, gradients_list, verbose=False):
        mean_grads = []
        for grads in zip(*gradients_list):
            grad_list = []
            for grad in grads:
                if grad is not None:
                    grad_list.append(grad)
            if len(grad_list) > 0:
                assert (len(grad_list) == len(grads))
                grads = tf.stack(grad_list, axis=0, name="stacked_gradients")
                if verbose:
                    print("stacked grads:", grads.shape)
                grad = tf.reduce_mean(grads, axis=0, name="mean_gradients")
                if verbose:
                    print("averaged grads:", grads.shape)
            else:
                grad = None
            mean_grads.append(grad)
        return mean_grads

    @property
    def gradients(self):
        return self._gradients

    @property
    def modules(self):
        return self._models[0].modules

    @property
    def loss(self):
        return self._loss

    @property
    def loss_min(self):
        return self._loss_min

    @property
    def loss_max(self):
        return self._loss_max
