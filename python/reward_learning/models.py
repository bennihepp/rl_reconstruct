from __future__ import print_function

import types
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from pybh import tf_utils


def update_module_options_with_defaults(module_type, module_options):
    if module_type == "conv3d" or module_type == "conv2d":
        default_options = {}
        default_options["num_convs_per_block"] = 8
        default_options["initial_num_filters"] = 8
        default_options["filter_increase_per_block"] = 0
        default_options["filter_increase_within_block"] = 6
        default_options["maxpool_after_each_block"] = False
        default_options["max_num_blocks"] = -1
        default_options["max_output_grid_size"] = 8
        default_options["dropout_rate"] = 0.5
        default_options["add_bias"] = False
        default_options["use_batch_norm"] = False
        default_options["activation_fn"] = "relu"
    elif module_type == "regression":
        default_options = {}
        default_options["use_batch_norm"] = False
        default_options["num_units"] = "1024"
        default_options["activation_fn"] = "relu"
    elif module_type == "upsampling1d" or module_type == "upsampling2d" or module_type == "upsampling3d":
        default_options = {}
        default_options["num_convs_per_block"] = 4
        default_options["add_bias"] = True
        default_options["use_batch_norm"] = False
        default_options["filter_decrease_per_block"] = 8
        default_options["filter_decrease_within_block"] = 16
        default_options["activation_fn"] = "relu"
    else:
        default_options = {}
    for key in default_options:
        if key not in module_options:
            module_options[key] = default_options[key]
    return module_options


def get_default_model_config():
    config = {}
    config["loss_mode"] = "mean"
    config["modules"] = ["conv3d", "regression"]
    config["conv3d"] = {}
    config["conv3d"]["num_convs_per_block"] = 8
    config["conv3d"]["initial_num_filters"] = 8
    config["conv3d"]["filter_increase_per_block"] = 0
    config["conv3d"]["filter_increase_within_block"] = 6
    config["conv3d"]["maxpool_after_each_block"] = False
    config["conv3d"]["max_num_blocks"] = -1
    config["conv3d"]["max_output_grid_size"] = 8
    config["conv3d"]["dropout_rate"] = 0.5
    config["conv3d"]["add_bias"] = False
    config["conv3d"]["use_batch_norm"] = False
    config["conv3d"]["activation_fn"] = "relu"
    config["conv2d"] = {}
    config["conv2d"]["num_convs_per_block"] = 8
    config["conv2d"]["initial_num_filters"] = 8
    config["conv2d"]["filter_increase_per_block"] = 0
    config["conv2d"]["filter_increase_within_block"] = 6
    config["conv2d"]["maxpool_after_each_block"] = False
    config["conv2d"]["max_num_blocks"] = -1
    config["conv2d"]["max_output_grid_size"] = 8
    config["conv2d"]["dropout_rate"] = 0.5
    config["conv2d"]["add_bias"] = False
    config["conv2d"]["use_batch_norm"] = False
    config["conv2d"]["activation_fn"] = "relu"
    config["regression"] = {}
    config["regression"]["use_batch_norm"] = False
    config["regression"]["num_units"] = "1024"
    config["regression"]["activation_fn"] = "relu"
    config["upsampling2d"] = {}
    config["upsampling2d"]["num_convs_per_block"] = 4
    config["upsampling2d"]["add_bias"] = True
    config["upsampling2d"]["use_batch_norm"] = False
    config["upsampling2d"]["filter_decrease_per_block"] = 8
    config["upsampling2d"]["filter_decrease_within_block"] = 16
    config["upsampling2d"]["activation_fn"] = "relu"
    config["upsampling3d"] = {}
    config["upsampling3d"]["num_convs_per_block"] = 4
    config["upsampling3d"]["add_bias"] = True
    config["upsampling3d"]["use_batch_norm"] = False
    config["upsampling3d"]["filter_decrease_per_block"] = 8
    config["upsampling3d"]["filter_decrease_within_block"] = 16
    config["upsampling3d"]["activation_fn"] = "relu"
    return config


def report_model_size(model, logger=None):
    if logger is None:
        log_fn = print
    else:
        log_fn = logger.info

    # Print model size etc.
    model_size = 0
    model_bytes = 0
    regularized_model_size = 0
    regularized_model_bytes = 0
    model_grad_size = 0
    model_grad_bytes = 0
    for grad, var in model.gradients_and_variables:
        model_size += np.sum(tf_utils.tensor_size(var))
        model_bytes += np.sum(tf_utils.tensor_bytes(var))
        if grad is not None:
            model_grad_size += np.sum(tf_utils.tensor_size(grad))
            model_grad_bytes += np.sum(tf_utils.tensor_bytes(grad))
    for var in set(model.regularized_variables):
        regularized_model_size += np.sum(tf_utils.tensor_size(var))
        regularized_model_bytes += np.sum(tf_utils.tensor_bytes(var))
    log_fn("Model variables: {} ({} MB)".format(model_size, model_bytes / 1024. / 1024.))
    log_fn("Regularized model variables: {} ({} MB)".format(regularized_model_size,
                                                            regularized_model_bytes / 1024. / 1024.))
    log_fn("Model gradients: {} ({} MB)".format(model_grad_size, model_grad_bytes / 1024. / 1024.))
    for name, module in model.modules_with_names:
        module_size = np.sum([tf_utils.tensor_size(var) for var in module.variables])
        module_bytes = np.sum([tf_utils.tensor_bytes(var) for var in module.variables])
        log_fn("Module {} variables: {} ({} MB)".format(name, module_size, module_bytes / 1024. / 1024.))


def add_model_loss_summaries(summary_wrapper):
    summary_wrapper.append_with_placeholder("loss", tf.float32)
    summary_wrapper.append_with_placeholder("loss_min", tf.float32)
    summary_wrapper.append_with_placeholder("loss_max", tf.float32)
    summary_wrapper.append_with_placeholder("regularization", tf.float32)
    summary_wrapper.append_with_placeholder("unweighted_loss", tf.float32)
    summary_wrapper.append_with_placeholder("unregularized_loss", tf.float32)


class ModelModule(object):

    def __init__(self, config, input_tensor):
        self._config = config
        self._input = input_tensor
        self._trainable_variables = []
        self._local_variables = []
        self._global_variables = []
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
        if type(config_value) == str:
            # Config value is a string. Split it at commas.
            config_value = config_value.strip("[]").split(',')
        elif type(config_value) != list:
            # Config value might be a single number. Convert it to a list with one element.
            config_value = [config_value]
        config_list = [converter(x) for x in config_value]
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

    def _add_variables(self, scope=None):
        if scope is None:
            scope_name = self._scope.name
        elif type(scope) == str:
            scope_name = scope
        else:
            scope_name = scope.name
        self._add_trainable_variables(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name))
        self._add_global_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name))
        self._add_local_variables(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope_name))

    def _add_variables_from_sub_scope(self, sub_scope_name):
        with tf.variable_scope(sub_scope_name) as scope:
            self._add_variables(scope)

    def _add_trainable_variable(self, var):
        self._trainable_variables.append(var)

    def _add_trainable_variables(self, vars):
        self._trainable_variables.extend(vars)

    def _add_global_variable(self, var):
        self._global_variables.append(var)

    def _add_global_variables(self, vars):
        self._global_variables.extend(vars)

    def _add_local_variable(self, var):
        self._local_variables.append(var)

    def _add_local_variables(self, vars):
        self._local_variables.extend(vars)

    def _add_summary(self, name, tensor):
        assert(name not in self._summaries)
        self._summaries[name] = tensor

    def _add_summaries(self, summary_dict, scope=None):
        for name, tensor in summary_dict.items():
            if scope is not None:
                name = scope.name + "/" + name
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
        return self._trainable_variables

    @property
    def trainable_variables(self):
        return self._trainable_variables

    @property
    def global_variables(self):
        return self._global_variables

    @property
    def local_variables(self):
        return self._local_variables

    @property
    def summaries(self):
        return self._summaries

    @property
    def scope(self):
        return self._scope

    @property
    def output(self):
        return self._output


class ModelModuleContainer(ModelModule):

    def __init__(self, config, input_tensor):
        self._modules = []
        super(ModelModuleContainer, self).__init__(config, input_tensor)

    def _add_module(self, module):
        self._modules.append(module)
        self._add_variables(module.scope)
        self._add_summaries(module.summaries, scope=module.scope)

    @property
    def modules(self):
        return self._modules


class DropoutLayer(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(DropoutLayer, self).__init__(config, input)

        dropout_rate = self._get_config("dropout_rate", 0.0)

        x = input
        if verbose:
            print("--- DropoutLayer ---")
            print("  input:", x.shape)

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class Conv3DLayer(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Conv3DLayer, self).__init__(config, input)

        filter_size = self._get_config("filter_size", [3, 3, 3])
        stride = self._get_config("stride", [1, 1, 1])
        num_filters = self._get_config("num_filters", None)
        if num_filters is None:
            num_filters = input.shape[-1] + self.config["num_filters_increase"]
        dropout_rate = self._get_config("dropout_rate", 0.0)
        use_batch_norm = self._get_config("use_batch_norm", True)
        add_bias = self._get_config("add_bias", not use_batch_norm)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", False)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn", "relu"))

        if use_batch_norm:
            assert not add_bias

        x = input
        if verbose:
            print("--- Conv3DLayer ---")
            print("  input:", x.shape)

        x, x_pre_activation = tf_utils.conv3d(
            x, num_filters, filter_size, stride,
            activation_fn=activation_fn,
            add_bias=add_bias,
            use_batch_norm=use_batch_norm,
            batch_norm_after_activation=batch_norm_after_activation,
            is_training=is_training,
            name="conv3d",
            variables_on_cpu=variables_on_cpu,
            collections=variables_collections,
            regularized_variables_collection=regularized_variables_collection,
            return_pre_activation=True)
        self._add_summary("conv3d/pre_activations", x_pre_activation)
        self._add_summary("conv3d/activations", x)
        self._add_variables_from_sub_scope("conv3d")

        if verbose:
            print("  x.shape:", x.shape)

        # Make layer accessible from outside
        with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
            weights = tf.get_variable("conv3d/weights")
            if use_batch_norm:
                biases = tf.get_variable("conv3d/bn/beta")
            elif add_bias:
                biases = tf.get_variable("conv3d/biases")
            else:
                biases = None
            # Add summaries for layer
            self._add_summary("conv3d/weights", weights)
            if biases is not None:
                self._add_summary("conv3d/biases", biases)

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class Conv3DResidualLayer(ModelModuleContainer):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Conv3DResidualLayer, self).__init__(config, input)

        dropout_rate = self._get_config("dropout_rate", 0.0)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        use_bottleneck = self._get_config("use_bottleneck", False)
        bottleneck_num_filters = self._get_config("bottleneck_num_filters", None)

        x = input
        if verbose:
            print("--- Conv3DResidualLayer ---")
            print("  input:", x.shape)

        if use_bottleneck:
            config1 = dict(config)
            config1["num_filters"] = config["num_filters"]
            config1["filter_size"] = [1, 1, 1]
            with tf.variable_scope("res_1"):
                module1 = Conv3DLayer(config1, input, is_training, variables_on_cpu, variables_collections,
                                      regularized_variables_collection, verbose)
            config2 = dict(config)
            if bottleneck_num_filters is None:
                bottleneck_num_filters = config["num_filters"] // 4
            config2["num_filters"] = bottleneck_num_filters
            with tf.variable_scope("res_2"):
                module2 = Conv3DLayer(config2, module1.output, is_training, variables_on_cpu, variables_collections,
                                      regularized_variables_collection, verbose)

            config3 = dict(config)
            config3["num_filters"] = input.shape[-1]
            config3["filter_size"] = [1, 1, 1]
            config3["activation_fn"] = "none"
            with tf.variable_scope("res_3"):
                module3 = Conv3DLayer(config3, module2.output, is_training, variables_on_cpu, variables_collections,
                                      regularized_variables_collection, verbose)

            self._add_module(module1)
            self._add_module(module2)
            self._add_module(module3)
            last_module = module3
        else:
            config1 = dict(config)
            config1["num_filters"] = input.shape[-1]
            with tf.variable_scope("res_1"):
                module1 = Conv3DLayer(config1, input, is_training, variables_on_cpu, variables_collections,
                                      regularized_variables_collection, verbose)

            config2 = dict(config)
            config2["num_filters"] = input.shape[-1]
            config2["activation_fn"] = "none"
            with tf.variable_scope("res_2"):
                module2 = Conv3DLayer(config2, module1.output, is_training, variables_on_cpu, variables_collections,
                                      regularized_variables_collection, verbose)

            self._add_module(module1)
            self._add_module(module2)
            last_module = module2

        x = activation_fn(tf.add(last_module.output, input, name="residual_add"), name="residual_activation")
        self._add_summary("residual_activation", x)

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class Shrink3DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Shrink3DModule, self).__init__(config, input)

        with_maxpool = self._get_config("shrink_with_maxpool", False)
        filter_size = self._get_config("shrink_filter_size", [2, 2, 2])
        stride = self._get_config("shrink_stride", [2, 2, 2])
        batch_norm_after_activation = self._get_config("batch_norm_after_shrink_conv", True)
        dropout_rate = self._get_config("dropout_rate", 0.0)
        use_batch_norm = self._get_config("use_batch_norm", False)
        activation_fn = self._get_config("activation_fn", "none")

        x = input
        if verbose:
            print("--- Shrink3DModule ---")
            print("  input:", x.shape)

        num_filters = input.shape[-1]
        if with_maxpool:
            with tf.variable_scope("shrink3d"):
                x, x_pre_activation = tf_utils.maxpool3d(
                    x, num_filters, filter_size, stride,
                    activation_fn=activation_fn,
                    add_bias=False,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="shrink3d",
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                self._add_summary("conv3d/pre_activations", x_pre_activation)
                self._add_summary("shrink3d/activations", x)
                self._add_variables_from_sub_scope("shrink3d")
        else:
            with tf.variable_scope("shrink3d"):
                x, x_pre_activation = tf_utils.conv3d(
                    x, num_filters, filter_size, stride,
                    activation_fn=None,
                    add_bias=False,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="shrink3d",
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                self._add_summary("conv3d/pre_activations", x_pre_activation)
                self._add_summary("shrink3d/activations", x)
                self._add_variables_from_sub_scope("shrink3d")

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class Conv3DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Conv3DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 2)
        filter_size = self._get_config("filter_size", [3, 3, 3])
        stride = self._get_config("stride", [1, 1, 1])
        shrink_after_each_block = self._get_config("shrink_after_each_block", True)
        shrink_with_maxpool = self._get_config("shrink_with_maxpool", False)
        shrink_filter_size = self._get_config("shrink_filter_size", [2, 2, 2])
        shrink_stride = self._get_config("shrink_stride", [2, 2, 2])
        batch_norm_after_shrink_conv = self._get_config("batch_norm_after_shrink_conv", True)
        initial_num_filters = self._get_config("initial_num_filters", 8)
        filter_increase_per_block = self._get_config("filter_increase_per_block", 8)
        filter_increase_within_block = self._get_config("filter_increase_within_block", 0)
        max_num_blocks = self._get_config("max_num_blocks", -1)
        max_output_grid_size = self._get_config("max_output_grid_size", 8)
        dropout_rate = self._get_config("dropout_rate", 0.3)
        use_residual_units = self._get_config("use_residual_units", False)
        use_batch_norm = self._get_config("use_batch_norm", True)
        add_bias = self._get_config("add_bias", not use_batch_norm)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        if use_batch_norm:
            assert not add_bias

        if not shrink_after_each_block:
            assert(max_num_blocks > 0)

        if max_num_blocks <= 0:
            assert(max_output_grid_size > 0)

        x = input
        if verbose:
            print("--- Conv3DModule ---")
            print("  input:", x.shape)

        if use_residual_units:
            prev_layer_1 = None
            prev_layer_2 = None

        num_filters = initial_num_filters
        i = 0
        done = False
        while not done:
            for j in range(num_convs_per_block):
                local_activation_fn = activation_fn
                if use_residual_units:
                    if j % 2 == 0 and j >= 2 is not None:
                        local_activation_fn = lambda y, **kwargs: activation_fn(tf.add(x, prev_layer_2), **kwargs)
                x, x_pre_activation = tf_utils.conv3d(
                    x, num_filters, filter_size, stride,
                    activation_fn=local_activation_fn,
                    add_bias=add_bias,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="conv3d_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if (j + 1) < num_convs_per_block:
                    num_filters += filter_increase_within_block
                self._add_summary("conv3d_{}_{}/pre_activations".format(i, j), x_pre_activation)
                self._add_summary("conv3d_{}_{}/activations".format(i, j), x)
                self._add_variables_from_sub_scope("conv3d_{}_{}".format(i, j))
                if use_residual_units:
                    prev_layer_2 = prev_layer_1
                    prev_layer_1 = x

            do_shrink = shrink_after_each_block or (max_num_blocks > 0 and (i + 1) >= max_num_blocks)
            if do_shrink:
                grid_size = max([int(x.shape[s]) for s in [-2, -3, -4]])
                if grid_size > max_output_grid_size:
                    if shrink_with_maxpool:
                        # with tf.variable_scope("maxpool3d_{}".format(i)):
                        #     x = tf.nn.max_pool3d(x, filter_size, stride, padding="SAME", data_format="NDHWC")
                        with tf.variable_scope("maxpool3d_{}".format(i)):
                            x, x_pre_activation = tf_utils.maxpool3d(
                                x, num_filters, shrink_filter_size, shrink_stride,
                                activation_fn=None,
                                add_bias=False,
                                use_batch_norm=batch_norm_after_shrink_conv,
                                batch_norm_after_activation=batch_norm_after_activation,
                                is_training=is_training,
                                name="shrink3d_{}".format(i),
                                variables_on_cpu=variables_on_cpu,
                                collections=variables_collections,
                                regularized_variables_collection=regularized_variables_collection,
                                return_pre_activation=True)
                            self._add_summary("conv3d_{}_maxpool/pre_activations".format(i), x_pre_activation)
                            self._add_summary("conv3d_{}_maxpool/activations".format(i), x)
                            self._add_variables_from_sub_scope("shrink3d_{}".format(i))
                    else:
                        with tf.variable_scope("shrink3d_{}".format(i)):
                            x, x_pre_activation = tf_utils.conv3d(
                                x, num_filters, shrink_filter_size, shrink_stride,
                                activation_fn=None,
                                add_bias=False,
                                use_batch_norm=batch_norm_after_shrink_conv,
                                batch_norm_after_activation=batch_norm_after_activation,
                                is_training=is_training,
                                name="shrink3d_{}".format(i),
                                variables_on_cpu=variables_on_cpu,
                                collections=variables_collections,
                                regularized_variables_collection=regularized_variables_collection,
                                return_pre_activation=True)
                            self._add_summary("conv3d_{}_shrink/pre_activations".format(i), x_pre_activation)
                            self._add_summary("conv3d_{}_shrink/activations".format(i), x)
                            self._add_variables_from_sub_scope("shrink3d_{}".format(i))
                elif max_output_grid_size > 0:
                    done = True
            if verbose:
                print("  x.shape:", i, x.shape)
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in range(num_convs_per_block):
                    weights = tf.get_variable("conv3d_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv3d_{}_{}/bn/beta".format(i, j))
                    elif add_bias:
                        biases = tf.get_variable("conv3d_{}_{}/biases".format(i, j))
                    else:
                        biases = None
                    # Add summaries for layer
                    self._add_summary("conv3d_{}_{}/weights".format(i, j), weights)
                    if biases is not None:
                        self._add_summary("conv3d_{}_{}/biases".format(i, j), biases)

            num_filters += filter_increase_per_block

            if max_num_blocks > 0 and (i + 1) >= max_num_blocks:
                done = True
            i += 1

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class Conv2DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Conv2DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 2)
        maxpool_after_each_block = self._get_config("maxpool_after_each_block", True)
        initial_num_filters = self._get_config("initial_num_filters", 8)
        filter_increase_per_block = self._get_config("filter_increase_per_block", 8)
        filter_increase_within_block = self._get_config("filter_increase_within_block", 0)
        max_num_blocks = self._get_config("max_num_blocks", -1)
        max_output_grid_size = self._get_config("max_output_grid_size", 8)
        dropout_rate = self._get_config("dropout_rate", 0.3)
        use_batch_norm = self._get_config("use_batch_norm", True)
        add_bias = self._get_config("add_bias", not use_batch_norm)
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
            print("--- Conv2DModule ---")
            print("  input:", x.shape)

        num_filters = initial_num_filters
        i = 0
        done = False
        while not done:
            filter_size = [3, 3]
            stride = [1, 1]
            for j in range(num_convs_per_block):
                x, x_pre_activation = tf_utils.conv2d(
                    x, num_filters, filter_size, stride,
                    activation_fn=activation_fn,
                    add_bias=add_bias,
                    use_batch_norm=use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    name="conv2d_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if (j + 1) < num_convs_per_block:
                    num_filters += filter_increase_within_block
                self._add_summary("conv2d_{}_{}/pre_activations".format(i, j), x_pre_activation)
                self._add_summary("conv2d_{}_{}/activations".format(i, j), x)
                self._add_variables_from_sub_scope("conv2d_{}_{}".format(i, j))
            do_maxpool = maxpool_after_each_block or (max_num_blocks > 0 and (i + 1) >= max_num_blocks)
            if do_maxpool:
                x_grid_size = int(x.shape[-2])
                if x_grid_size > max_output_grid_size:
                    ksize = [1, 2, 2, 1]
                    strides = [1, 2, 2, 1]
                    with tf.variable_scope("maxpool2d_{}".format(i)):
                        x = tf.nn.max_pool(x, ksize, strides, padding="SAME", data_format="NHWC")
                elif max_output_grid_size > 0:
                    done = True
            if verbose:
                print("  x.shape:", i, x.shape)
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in range(num_convs_per_block):
                    weights = tf.get_variable("conv2d_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv2d_{}_{}/bn/beta".format(i, j))
                    elif add_bias:
                        biases = tf.get_variable("conv2d_{}_{}/biases".format(i, j))
                    else:
                        biases = None
                    # Add summaries for layer
                    self._add_summary("conv2d_{}_{}/weights".format(i, j), weights)
                    if biases is not None:
                        self._add_summary("conv2d_{}_{}/biases".format(i, j), biases)

            num_filters += filter_increase_per_block

            if max_num_blocks > 0 and (i + 1) >= max_num_blocks:
                done = True
            i += 1

        if dropout_rate > 0 and is_training:
            keep_prob = 1 - dropout_rate
            noise_shape = [tf.shape(x)[0], 1, 1, tf.shape(x)[-1]]
            x = tf.nn.dropout(x, keep_prob=keep_prob,
                              noise_shape=noise_shape, name="dropout")
            if verbose:
                print("  x.shape, dropout:", x.shape)
            self._add_summary("dropout_activations", x)

        self._set_output(x)


class FullyConnectedModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(FullyConnectedModule, self).__init__(config, input)

        num_units = self._get_config_int_list("num_units")
        assert(num_units is not None)
        use_batch_norm = self._get_config("use_batch_norm", True)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = self._get_config("add_bias")
        fully_convolutional = self._get_config("fully_convolutional", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        if use_batch_norm:
            assert not add_bias

        input_shape = input.get_shape().as_list()
        x = input
        if verbose:
            print("--- FullyConnected ---")
            print("  input:", x.shape)

        def regression_layer(_x, _num_units, _activation_fn, _add_bias, _use_batch_norm):
            if fully_convolutional:
                if len(input_shape) == 5:
                    filter_size = [int(s) for s in x.shape[1:-1]]
                    stride = [1, 1, 1]
                    _x, _x_pre_activation = tf_utils.conv3d(
                        _x, _num_units, filter_size, stride,
                        activation_fn=_activation_fn,
                        add_bias=_add_bias,
                        use_batch_norm=_use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        padding="VALID",
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections,
                        regularized_variables_collection=regularized_variables_collection,
                        return_pre_activation=True)
                elif len(input_shape) == 4:
                    filter_size = [int(s) for s in x.shape[1:-1]]
                    stride = [1, 1]
                    _x, _x_pre_activation = tf_utils.conv2d(
                        _x, _num_units, filter_size, stride,
                        activation_fn=_activation_fn,
                        add_bias=_add_bias,
                        use_batch_norm=_use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        padding="VALID",
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections,
                        regularized_variables_collection=regularized_variables_collection,
                        return_pre_activation=True)
                else:
                    raise RuntimeError("Only inputs of rank 4 or 5 are supported for"
                                       "fully-convolutional regression module")
            else:
                _x, _x_pre_activation = tf_utils.fully_connected(
                    _x, _num_units,
                    activation_fn=_activation_fn,
                    add_bias=_add_bias,
                    use_batch_norm=_use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if verbose:
                    print("  activations.shape, fc_{}:".format(i), x.shape)
            return _x, _x_pre_activation

        assert(len(num_units) > 0)
        for i in range(len(num_units)):
            with tf.variable_scope('fc_{}'.format(i)) as scope:
                x, x_pre_activation = regression_layer(x, num_units[i], activation_fn, add_bias, use_batch_norm)
                if verbose:
                    print("  activations.shape, fc_{}:".format(i), x.shape)
                self._add_summary("fc_{}/pre_activations".format(i), x_pre_activation)
                self._add_summary("fc_{}/activations".format(i), x)
            with tf.variable_scope(scope, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                if verbose:
                    print("  {}: {}".format(weights.name, weights.shape))
                if use_batch_norm:
                    biases = tf.get_variable("bn/beta")
                elif add_bias:
                    biases = tf.get_variable('biases')
                else:
                    biases = None
                if verbose and biases is not None:
                    print("  {}: {}".format(biases.name, biases.shape))
                # Add summaries for layer
                self._add_summary("fc_{}/weights".format(i), weights)
                if biases is not None:
                    self._add_summary("fc_{}/biases".format(i), biases)
            self._add_variables(scope)

        self._set_output(x)


class LambdaModule(ModelModule):

    def __init__(self, input, lambda_fn, verbose=False):
        config = {}
        super(LambdaModule, self).__init__(config, input)

        x = input
        if verbose:
            print("--- LambdaModule ---")
            print("  input_shape:", x.shape)

        x = lambda_fn(x)

        self._set_output(x)


class ReshapeModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(ReshapeModule, self).__init__(config, input)

        output_shape = self._get_config_int_list("output_shape")
        assert(output_shape is not None)

        x = input
        if verbose:
            print("--- Reshape ---")
            print("  input_shape:", x.shape)

        x = tf.reshape(x, [int(x.shape[0])] +  output_shape)

        self._set_output(x)


class FlattenModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(FlattenModule, self).__init__(config, input)

        x = input
        if verbose:
            print("--- Flatten ---")
            print("  input_shape:", x.shape)

        x = tf.reshape(x, [int(x.shape[0]), -1])

        self._set_output(x)


class ExpandDimsModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(ExpandDimsModule, self).__init__(config, input)

        axis = self._get_config("axis")

        x = input
        if verbose:
            print("--- ExpandDims ---")
            print("  input_shape:", x.shape)

        x = tf.expand_dims(x, axis=axis)

        self._set_output(x)


class ScaleTensorModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(ScaleTensorModule, self).__init__(config, input)

        scale_factor = self._get_config("scale_factor", 1)
        offset = self._get_config("offset", 0)

        x = input
        if verbose:
            print("--- ScaleTensor ---")
            print("  input_shape:", x.shape)
            print("  scale_factor:", scale_factor)
            print("  offset:", offset)

        x = scale_factor * x + offset

        self._set_output(x)


class ClipTensorModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(ClipTensorModule, self).__init__(config, input)

        min_value = self._get_config("min", input.dtype.min)
        max_value = self._get_config("max", input.dtype.max)

        x = input
        if verbose:
            print("--- ClipTensor ---")
            print("  input_shape:", x.shape)
            print("  min_value:", min_value)
            print("  max_value:", max_value)

        x = tf.clip_by_value(input, min_value, max_value)

        self._set_output(x)


class SliceTensorModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(SliceTensorModule, self).__init__(config, input)

        slices_cfg = self._get_config("slices")
        slices = []
        # Batch dimension
        slices.append(slice(None))
        for slice_cfg in slices_cfg:
            if len(slice_cfg) == 0:
                slices.append(slice(None))
            else:
                slices.append(slice(slice_cfg[0], slice_cfg[1]))

        x = input
        if verbose:
            print("--- SliceTensor ---")
            print("  input_shape:", x.shape)

        x = x[slices]

        self._set_output(x)


class ResizeImagesModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 verbose=False):
        super(ResizeImagesModule, self).__init__(config, input)

        new_size = self._get_config_int_list("size")
        resize_method_str = self._get_config("method", "bilinear")
        if resize_method_str == "nearest":
            resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif resize_method_str == "bilinear":
            resize_method = tf.image.ResizeMethod.BILINEAR
        elif resize_method_str == "bicubic":
            resize_method = tf.image.ResizeMethod.BICUBIC
        elif resize_method_str == "area":
            resize_method = tf.image.ResizeMethod.AREA
        else:
            raise RuntimeError("Unknown image resize method: {}".format(resize_method_str))

        x = input
        if verbose:
            print("--- ResizeImages ---")
            print("  input_shape:", x.shape)

        x = tf.image.resize_images(x, new_size, resize_method)

        self._set_output(x)


class RegressionModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 num_outputs,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(RegressionModule, self).__init__(config, input)

        num_units = self._get_config_int_list("num_units")
        assert(num_units is not None)
        use_batch_norm = self._get_config("use_batch_norm", True)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = self._get_config("add_bias", not use_batch_norm)
        fully_convolutional = self._get_config("fully_convolutional", True)
        activation_fn = self._get_activation_fn(self._get_config("activation_fn"))

        if use_batch_norm:
            assert not add_bias

        input_shape = input.get_shape().as_list()
        x = input
        if verbose:
            print("--- RegressionModule ---")
            print("  input:", x.shape)

        def regression_layer(_x, _num_units, _activation_fn, _add_bias, _use_batch_norm):
            if fully_convolutional:
                if len(input_shape) == 5:
                    filter_size = [int(s) for s in x.shape[1:-1]]
                    stride = [1, 1, 1]
                    _x, _x_pre_activation = tf_utils.conv3d(
                        _x, _num_units, filter_size, stride,
                        activation_fn=_activation_fn,
                        add_bias=_add_bias,
                        use_batch_norm=_use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        padding="VALID",
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections,
                        regularized_variables_collection=regularized_variables_collection,
                        return_pre_activation=True)
                elif len(input_shape) == 4:
                    filter_size = [int(s) for s in x.shape[1:-1]]
                    stride = [1, 1]
                    _x, _x_pre_activation = tf_utils.conv2d(
                        _x, _num_units, filter_size, stride,
                        activation_fn=_activation_fn,
                        add_bias=_add_bias,
                        use_batch_norm=_use_batch_norm,
                        batch_norm_after_activation=batch_norm_after_activation,
                        is_training=is_training,
                        padding="VALID",
                        variables_on_cpu=variables_on_cpu,
                        collections=variables_collections,
                        regularized_variables_collection=regularized_variables_collection,
                        return_pre_activation=True)
                else:
                    raise RuntimeError("Only inputs of rank 4 or 5 are supported for"
                                       "fully-convolutional regression module")
            else:
                _x, _x_pre_activation = tf_utils.fully_connected(
                    _x, _num_units,
                    activation_fn=_activation_fn,
                    add_bias=_add_bias,
                    use_batch_norm=_use_batch_norm,
                    batch_norm_after_activation=batch_norm_after_activation,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if verbose:
                    print("  activations.shape, fc_{}:".format(i), x.shape)
            return _x, _x_pre_activation

        for i in range(len(num_units)):
            with tf.variable_scope('fc_{}'.format(i)) as scope:
                x, x_pre_activation = regression_layer(x, num_units[i], activation_fn, add_bias, use_batch_norm)
                if verbose:
                    print("  activations.shape, fc_{}:".format(i), x.shape)
                self._add_summary("fc_{}/pre_activations".format(i), x_pre_activation)
                self._add_summary("fc_{}/activations".format(i), x)
            with tf.variable_scope(scope, reuse=True):
                # Make layer accessible from outside
                weights = tf.get_variable('weights')
                if verbose:
                    print("  {}: {}".format(weights.name, weights.shape))
                if use_batch_norm:
                    biases = tf.get_variable("bn/beta")
                elif add_bias:
                    biases = tf.get_variable('biases')
                else:
                    biases = None
                if verbose and biases is not None:
                    print("  {}: {}".format(biases.name, biases.shape))
                # Add summaries for layer
                self._add_summary("fc_{}/weights".format(i), weights)
                if biases is not None:
                    self._add_summary("fc_{}/biases".format(i), biases)
            self._add_variables(scope)

        with tf.variable_scope('output') as scope:
            x, x_pre_activation = regression_layer(
                x, num_outputs, _activation_fn=None, _add_bias=True, _use_batch_norm=False)
            if fully_convolutional:
                if len(input_shape) == 5:
                    if int(x.shape[1]) == 1 and int(x.shape[2]) == 1 and int(x.shape[3]) == 1:
                        # Remove 3D tensor dimensions if there is no spatial extent.
                        # Otherwise reductions might lead to non-expected results.
                        x = tf_utils.reshape_batch_3d_to_2d(x)
                        x = tf_utils.reshape_batch_2d_to_1d(x)
                        x = tf_utils.reshape_batch_1d_to_flat(x)
                        # x = tf.reshape(x, [int(x.shape[0]), -1])
                elif len(input_shape) == 4:
                    if int(x.shape[1]) == 1 and int(x.shape[2]) == 1:
                        # Remove 3D tensor dimensions if there is no spatial extent.
                        # Otherwise reductions might lead to non-expected results.
                        x = tf_utils.reshape_batch_3d_to_2d(x)
                        x = tf_utils.reshape_batch_1d_to_flat(x)
                        # x = tf.reshape(x, [int(x.shape[0]), -1])
                else:
                    raise RuntimeError("Only inputs of rank 4 or 5 are supported for"
                                       "fully-convolutional regression module")
            self._add_summary("output/pre_activations", x_pre_activation)
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
            self._add_variables(scope)

        self._set_output(x)


class Upsampling1DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 target_shape,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Upsampling1DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 1)
        initial_num_filters = self._get_config("initial_num_filters", -1)
        filter_decrease_per_block = self._get_config("filter_decrease_per_block", 8)
        filter_decrease_within_block = self._get_config("filter_decrease_within_block", 0)
        use_batch_norm = self._get_config("use_batch_norm", False)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        activation_fn = self._get_activation_fn(self._get_config("activation_fn", None))
        use_activation_on_last_layer = self._get_config("use_activation_on_last_layer", False)

        if initial_num_filters <= 0:
            initial_num_filters = int(input.shape[-1])

        x = input
        if verbose:
            print("--- Upsampling1DModule ---")
            print("  input:", x.shape)

        i = 0
        num_filters = initial_num_filters
        last_conv_transpose = False
        while True:
            x_grid_size = int(x.shape[-2])
            if x_grid_size >= target_shape[-2]:
                break
            filter_size = [3]
            for j in range(num_convs_per_block):
                if (j + 1) < num_convs_per_block:
                    num_filters -= filter_decrease_within_block
                    output_shape = [int(s) for s in x.shape]
                    output_shape[-1] = num_filters
                    stride = [1]
                else:
                    output_shape = None
                    new_output_x_size = int(x.shape[-2]) * 2
                    stride = [2]
                    print("Last upsampling layer")
                    if (j + 1) >= num_convs_per_block and new_output_x_size >= target_shape[-2]:
                        last_conv_transpose = True
                        output_shape = [int(input.shape[0])] + target_shape
                        num_filters = target_shape[-1]
                    if last_conv_transpose and not use_activation_on_last_layer:
                        activation_fn = None
                x, x_pre_activation = tf_utils.conv1d_transpose(
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
                    name="conv1d_trans_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if last_conv_transpose:
                    for k in range(num_filters):
                        self._add_summary("output_{}".format(k), x[..., slice(k, k+1)])
                    self._add_summary("output_all", x)
                else:
                    self._add_summary("conv1d_trans_{}_{}/pre_activations".format(i, j), x_pre_activation)
                    self._add_summary("conv1d_trans_{}_{}/activations".format(i, j), x)
                self._add_variables_from_sub_scope("conv1d_trans_{}_{}".format(i, j))
                if verbose:
                    print("  x.shape, {} {}: {}".format(i, j, x.shape))
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in range(num_convs_per_block):
                    weights = tf.get_variable("conv1d_trans_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv1d_trans_{}_{}/bn/beta".format(i, j))
                    elif add_bias:
                        biases = tf.get_variable("conv1d_trans_{}_{}/biases".format(i, j))
                    else:
                        biases = None
                    # Add summaries for layer
                    self._add_summary("conv1d_trans_{}_{}/weights".format(i, j), weights)
                    if biases is not None:
                        self._add_summary("conv1d_trans_{}_{}/biases".format(i, j), biases)

            num_filters -= filter_decrease_per_block
            i += 1

        self._set_output(x)


class Upsampling2DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 target_shape,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Upsampling2DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 1)
        initial_num_filters = self._get_config("initial_num_filters", -1)
        filter_decrease_per_block = self._get_config("filter_decrease_per_block", 8)
        filter_decrease_within_block = self._get_config("filter_decrease_within_block", 0)
        use_batch_norm = self._get_config("use_batch_norm", False)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        activation_fn = self._get_activation_fn(self._get_config("activation_fn", None))
        use_activation_on_last_layer = self._get_config("use_activation_on_last_layer", False)

        if initial_num_filters <= 0:
            initial_num_filters = int(input.shape[-1])

        x = input
        if verbose:
            print("--- Upsampling2DModule ---")
            print("  input:", x.shape)

        i = 0
        num_filters = initial_num_filters
        last_conv_transpose = False
        while True:
            x_shape = x.get_shape().as_list()
            if np.all(x_shape[1:-1] >= target_shape[:-1]):
                break
            filter_size = [3, 3]
            for j in range(num_convs_per_block):
                if (j + 1) < num_convs_per_block:
                    num_filters -= filter_decrease_within_block
                    output_shape = [int(s) for s in x.shape]
                    output_shape[-1] = num_filters
                    stride = [1, 1]
                else:
                    x_shape = x.get_shape().as_list()
                    output_shape = [int(x_shape[0]), 2 * x_shape[1], 2 * x_shape[2], num_filters]
                    # print("  x:", x_shape)
                    # print("  output_shape before:", output_shape)
                    stride = [2, 2]
                    if output_shape[1] > target_shape[0]:
                        output_shape[1] = x_shape[1]
                        stride[0] = 1
                    if output_shape[2] > target_shape[1]:
                        output_shape[2] = x_shape[2]
                        stride[1] = 1
                    if verbose:
                        print("  output_shape:", output_shape)
                    print("Last upsampling layer")
                    if (j + 1) >= num_convs_per_block and np.all(output_shape[1:-1] >= target_shape[:-1]):
                        last_conv_transpose = True
                        num_filters = target_shape[-1]
                        output_shape[-1] = num_filters
                    if last_conv_transpose and not use_activation_on_last_layer:
                        activation_fn = None
                print("  num_filters", num_filters)
                print("  output_shape", output_shape)
                x, x_pre_activation = tf_utils.conv2d_transpose(
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
                    name="conv2d_trans_{}_{}".format(i, j),
                    variables_on_cpu=variables_on_cpu,
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if last_conv_transpose:
                    for k in range(num_filters):
                        self._add_summary("output_{}".format(k), x[..., slice(k, k+1)])
                    self._add_summary("output_all", x)
                else:
                    self._add_summary("conv2d_trans_{}_{}/pre_activations".format(i, j), x_pre_activation)
                    self._add_summary("conv2d_trans_{}_{}/activations".format(i, j), x)
                self._add_variables_from_sub_scope("conv2d_trans_{}_{}".format(i, j))
                if verbose:
                    print("  x.shape, {} {}: {}".format(i, j, x.shape))
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in range(num_convs_per_block):
                    weights = tf.get_variable("conv2d_trans_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv2d_trans_{}_{}/bn/beta".format(i, j))
                    elif add_bias:
                        biases = tf.get_variable("conv2d_trans_{}_{}/biases".format(i, j))
                    else:
                        biases = None
                    # Add summaries for layer
                    self._add_summary("conv2d_trans_{}_{}/weights".format(i, j), weights)
                    if biases is not None:
                        self._add_summary("conv2d_trans_{}_{}/biases".format(i, j), biases)

            num_filters -= filter_decrease_per_block
            i += 1

        self._set_output(x)


class Upsampling3DModule(ModelModule):

    def __init__(self,
                 config,
                 input,
                 target_shape,
                 is_training=True,
                 variables_on_cpu=False,
                 variables_collections=None,
                 regularized_variables_collection=None,
                 verbose=False):
        super(Upsampling3DModule, self).__init__(config, input)

        num_convs_per_block = self._get_config("num_convs_per_block", 1)
        initial_num_filters = self._get_config("initial_num_filters", -1)
        filter_decrease_per_block = self._get_config("filter_decrease_per_block", 8)
        filter_decrease_within_block = self._get_config("filter_decrease_within_block", 0)
        use_batch_norm = self._get_config("use_batch_norm", False)
        batch_norm_after_activation = self._get_config("batch_norm_after_activation", True)
        add_bias = not use_batch_norm
        activation_fn = self._get_activation_fn(self._get_config("activation_fn", None))
        use_activation_on_last_layer = self._get_config("use_activation_on_last_layer", False)

        if initial_num_filters <= 0:
            initial_num_filters = int(input.shape[-1])

        x = input
        if verbose:
            print("--- Upsampling3DModule ---")
            print("  input:", x.shape)

        i = 0
        num_filters = initial_num_filters
        last_conv_transpose = False
        while True:
            x_grid_size = int(x.shape[-4])
            if x_grid_size >= target_shape[-4]:
                break
            filter_size = [3, 3, 3]
            for j in range(num_convs_per_block):
                if (j + 1) < num_convs_per_block:
                    num_filters -= filter_decrease_within_block
                    output_shape = [int(s) for s in x.shape]
                    output_shape[-1] = num_filters
                    stride = [1, 1, 1]
                else:
                    output_shape = None
                    new_output_x_size = int(x.shape[-4]) * 2
                    stride = [2, 2, 2]
                    print("Last upsampling layer")
                    if (j + 1) >= num_convs_per_block and new_output_x_size >= target_shape[-4]:
                        last_conv_transpose = True
                        num_filters = target_shape[-1]
                    if last_conv_transpose and not use_activation_on_last_layer:
                        activation_fn = None
                x, x_pre_activation = tf_utils.conv3d_transpose(
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
                    collections=variables_collections,
                    regularized_variables_collection=regularized_variables_collection,
                    return_pre_activation=True)
                if last_conv_transpose:
                    for k in range(num_filters):
                        self._add_summary("output_{}".format(k), x[..., slice(k, k+1)])
                    self._add_summary("output_all", x)
                else:
                    self._add_summary("conv3d_trans_{}_{}/pre_activations".format(i, j), x_pre_activation)
                    self._add_summary("conv3d_trans_{}_{}/activations".format(i, j), x)
                self._add_variables_from_sub_scope("conv3d_trans_{}_{}".format(i, j))
                if verbose:
                    print("  x.shape, {} {}: {}".format(i, j, x.shape))
            # Make layer accessible from outside
            with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
                for j in range(num_convs_per_block):
                    weights = tf.get_variable("conv3d_trans_{}_{}/weights".format(i, j))
                    if use_batch_norm:
                        biases = tf.get_variable("conv3d_trans_{}_{}/bn/beta".format(i, j))
                    elif add_bias:
                        biases = tf.get_variable("conv3d_trans_{}_{}/biases".format(i, j))
                    else:
                        biases = None
                    # Add summaries for layer
                    self._add_summary("conv3d_trans_{}_{}/weights".format(i, j), weights)
                    if biases is not None:
                        self._add_summary("conv3d_trans_{}_{}/biases".format(i, j), biases)

            num_filters -= filter_decrease_per_block
            i += 1

        self._set_output(x)


class Model(ModelModule):

    def _convert_config_old_to_new(self, config):
        if "modules" not in config:
            modules_cfg = []
            if "conv2d" in config:
                modules_cfg.append({
                    "type": "conv2d_module",
                    "options": config["conv2d"]})
            elif "conv3d" in config:
                modules_cfg.append({
                    "type": "conv3d_module",
                    "options": config["conv3d"]})
            if "regression" in config:
                modules_cfg.append({
                    "type": "regression_module",
                    "options": config["regression"]})
            elif "upsampling2d" in config:
                modules_cfg.append({
                    "type": "upsampling2d_module",
                    "options": config["upsampling2d"]})
            elif "upsampling3d" in config:
                modules_cfg.append({
                    "type": "upsampling3d_module",
                    "options": config["upsampling3d"]})
            config["modules"] = modules_cfg

        if "loss" not in config:
            loss_cfg = {}
            if "loss_mode" in config:
                loss_cfg["reduce_mode"] = config["loss_mode"]
            if "regularization_mode" in config:
                loss_cfg["regularization_mode"] = config["regularization_mode"]
            if "regularization_lambda" in config:
                loss_cfg["regularization_lambda"] = config["regularization_lambda"]
            config["loss"] = loss_cfg

        return config

    def __init__(self, config, input_batch, target_batch, weight_batch=None, is_training=True,
                 variables_on_cpu=False, variables_collections=None,
                 regularized_variables_collection=None, verbose=False):
        # Convert config from old to new format
        config = self._convert_config_old_to_new(config)

        super(Model, self).__init__(config, input_batch)

        if regularized_variables_collection is None:
            regularized_variables_collection = "__RegularizedVariables__"
        self._regularized_variables_collection = regularized_variables_collection

        target_shape = [int(s) for s in target_batch.shape[1:]]

        loss_cfg = self._get_config("loss", {})

        loss_type = loss_cfg.get("type", "l2_norm")
        if loss_type == "l2_norm":
            loss_fn = tf_utils.loss_l2_norm
        elif loss_type == "l1_norm":
            loss_fn = tf_utils.loss_l1_norm
        elif loss_type == "neg_log_likelihood_normal":
            def neg_log_likelihood_normal_loss_fn(target, prediction):
                mean_prediction = tf.expand_dims(prediction[..., 0], axis=-1)
                log_sigma_prediction = tf.expand_dims(prediction[..., 1], axis=-1)
                sigma_prediction = tf.exp(log_sigma_prediction)
                log_likelihood = tf_utils.log_likelihood_normal(target, mean_prediction, sigma_prediction)
                neg_log_likelihood = tf.negative(log_likelihood)
                return neg_log_likelihood
            loss_fn = neg_log_likelihood_normal_loss_fn
        else:
            raise NotImplementedError("Unknown loss type: {}".format(loss_type))

        loss_reduce_mode = loss_cfg.get("reduce_mode", "mean")
        if loss_reduce_mode == "mean":
            loss_reduce_fn = tf.reduce_mean
        elif loss_reduce_mode == "sum":
            loss_reduce_fn = tf.reduce_sum
        else:
            raise NotImplementedError("Unknown loss reduce mode: {}".format(loss_reduce_mode))

        regularization_mode = loss_cfg.get("regularization_mode", "none")
        if regularization_mode == "none":
            regularization_fn = None
        elif regularization_mode == "l2_norm":
            regularization_fn = lambda x: tf.reduce_sum(tf.square(x), axis=-1, name="regularization_batch")
        elif regularization_mode == "l1_norm":
            regularization_fn = lambda x: tf.reduce_sum(tf.abs(x), axis=-1, name="regularization_batch")
        else:
            raise NotImplementedError("Unknown regularization mode: {}".format(regularization_mode))
        if regularization_fn is not None:
            regularization_lambda = float(loss_cfg.get("regularization_lambda", None))
            if regularization_lambda is None:
                raise RuntimeError("Regularization parameter lambda has to be provided")

        x_batch = input_batch

        self._modules = []
        self._module_names = []

        modules_cfg = self._get_config("modules", [])
        assert(len(modules_cfg) > 0)
        for i, module_cfg in enumerate(modules_cfg):
            module_type = module_cfg["type"]
            module_name = module_cfg.get("name", "{}_{}".format(module_type, i))
            module_options = module_cfg.get("options", {})
            update_module_options_with_defaults(module_type, module_options)

            if module_type == "conv3d_layer":
                with tf.variable_scope(module_name):
                    module = Conv3DLayer(
                        module_options,
                        x_batch,
                        is_training=is_training,
                        variables_on_cpu=variables_on_cpu,
                        regularized_variables_collection=self._regularized_variables_collection,
                        variables_collections=variables_collections,
                        verbose=verbose)

            elif module_type == "conv3d_residual_layer":
                with tf.variable_scope(module_name):
                    module = Conv3DResidualLayer(
                        module_options,
                        x_batch,
                        is_training=is_training,
                        variables_on_cpu=variables_on_cpu,
                        regularized_variables_collection=self._regularized_variables_collection,
                        variables_collections=variables_collections,
                        verbose=verbose)

            elif module_type == "dropout_layer":
                with tf.variable_scope(module_name):
                    module = DropoutLayer(
                        module_options,
                        x_batch,
                        is_training=is_training,
                        variables_on_cpu=variables_on_cpu,
                        regularized_variables_collection=self._regularized_variables_collection,
                        variables_collections=variables_collections,
                        verbose=verbose)

            elif module_type == "shrink3d":
                with tf.variable_scope(module_name):
                    module = Shrink3DModule(
                        module_options,
                        x_batch,
                        is_training=is_training,
                        variables_on_cpu=variables_on_cpu,
                        regularized_variables_collection=self._regularized_variables_collection,
                        variables_collections=variables_collections,
                        verbose=verbose)

            elif module_type == "conv3d_module":
                module = Conv3DModule(
                    module_options,
                    x_batch,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "conv2d_module":
                x_shape = x_batch.get_shape().as_list()
                assert(x_shape[-2] == 1)
                new_x_shape = x_shape[:3] + [x_shape[-1]]
                x_batch = tf.reshape(x_batch, new_x_shape)
                module = Conv2DModule(
                    module_options,
                    x_batch,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "fully_connected_module":
                module = FullyConnectedModule(
                    module_options,
                    x_batch,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "regression_module":
                assert(i == len(modules_cfg) - 1)
                assert(len(target_shape) == 0 or len(target_shape) == 1)
                if len(target_shape) == 0:
                    num_outputs = 1
                else:
                    num_outputs = target_shape[-1]
                module = RegressionModule(
                    module_options,
                    x_batch,
                    num_outputs,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "upsampling1d_module":
                module_target_shape = module_options.get("target_shape", target_shape)
                if len(module_target_shape) == 3 and module_target_shape[-3] == 1:
                    module_target_shape = module_target_shape[-2:]
                elif len(module_target_shape) == 4 and module_target_shape[-4] == 1:
                    module_target_shape = module_target_shape[-3:]
                module = Upsampling1DModule(
                    module_options,
                    x_batch,
                    module_target_shape,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "upsampling2d_module":
                module_target_shape = module_options.get("target_shape", target_shape)
                module = Upsampling2DModule(
                    module_options,
                    x_batch,
                    module_target_shape,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "upsampling3d_module":
                module_target_shape = module_options.get("target_shape", target_shape)
                module = Upsampling3DModule(
                    module_options,
                    x_batch,
                    module_target_shape,
                    is_training=is_training,
                    variables_on_cpu=variables_on_cpu,
                    regularized_variables_collection=self._regularized_variables_collection,
                    variables_collections=variables_collections,
                    verbose=verbose)

            elif module_type == "reshape":
                module = ReshapeModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "flatten":
                module = FlattenModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "expand_dims":
                module = ExpandDimsModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "flat_to_1d":
                module = LambdaModule(
                    x_batch,
                    lambda x: tf_utils.reshape_batch_1d_to_2d(x),
                    verbose=verbose)

            elif module_type == "1d_to_2d":
                module = LambdaModule(
                    x_batch,
                    lambda x: tf_utils.reshape_batch_1d_to_2d(x),
                    verbose = verbose)

            elif module_type == "2d_to_3d":
                module = LambdaModule(
                    x_batch,
                    lambda x: tf_utils.reshape_batch_1d_to_2d(x),
                    verbose=verbose)

            elif module_type == "flatten":
                module = FlattenModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "resize_images":
                module = ResizeImagesModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "scale":
                module = ScaleTensorModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "clip":
                module = ClipTensorModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            elif module_type == "slice":
                module = SliceTensorModule(
                    module_options,
                    x_batch,
                    verbose=verbose)

            else:
                raise RuntimeError("Unknown module type: {}".format(module_type))

            print("Module {} of type {} has output shape {}".format(module_name, module_type, module.output.shape))

            x_batch = module.output
            self._module_names.append(module_name)
            self._modules.append(module)

        self._set_output(self._modules[-1].output)

        self.qwe1 = target_batch
        self.qwe2 = self.output
        self.abc = tf.reduce_mean(tf_utils.loss_l2_norm(target_batch, self.output))

        # Flatten output and target to compute loss
        if True:
            print("Shape of output: {}".format(self.output.shape))
            print("Shape of target_batch: {}".format(target_batch.shape))
        # reshaped_output = tf.reshape(self.output, [int(self.output.shape[0]), -1])
        # reshaped_target_batch = tf.reshape(target_batch, [int(self.output.shape[0]), -1])
        # if True:
        #     print("Shape of reshaped output: {}".format(reshaped_output.shape))
        #     print("Shape of reshaped target_batch: {}".format(reshaped_target_batch.shape))

        tmp_loss_batch = loss_fn(target_batch, self.output)
        while tmp_loss_batch.get_shape().ndims > 2:
            print("Fusing dimension of loss. Current shape: {}".format(tmp_loss_batch.get_shape().as_list()))
            tmp_loss_batch = tf_utils.reshape_fuse_dimension(tmp_loss_batch, -3)
            print("New shape: {}".format(tmp_loss_batch.get_shape().as_list()))
        self._unweighted_loss_batch = loss_reduce_fn(tmp_loss_batch, axis=-1, name="unweighted_loss_batch")
        if verbose:
            print("Shape of batch unweighted loss: {}".format(self._unweighted_loss_batch.shape))
        if weight_batch is None:
            self._unregularized_loss_batch = tf.identity(self._unweighted_loss_batch, name="unregularized_loss_batch")
        else:
            self._unregularized_loss_batch = tf.multiply(weight_batch, self._unweighted_loss_batch, name="unregularized_loss_batch")
        if verbose:
            print("Shape of batch unregularized loss: {}".format(self._unregularized_loss_batch.shape))

        if regularization_fn is None:
            self.__regularization_batch = None
        else:
            if self._regularized_variables_collection is None:
                regularized_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                regularized_variables = tf.get_collection(self._regularized_variables_collection)
                regularized_variables = [tf_utils.flatten_tensor(x) for x in regularized_variables]
                regularized_variables = tf.concat(regularized_variables, axis=0)
            self.__regularization_batch = regularization_fn(regularized_variables)

        for module_name, module in zip(self._module_names, self._modules):
            self._add_trainable_variables(module.trainable_variables)
            self._add_global_variables(module.global_variables)
            self._add_local_variables(module.local_variables)
            for tensor_name, tensor in module.summaries.items():
                self._add_summary(module_name + "/" + tensor_name, tensor)

        self._unweighted_loss = tf.reduce_mean(self._unweighted_loss_batch, name="unweighted_loss")
        self._unregularized_loss = tf.reduce_mean(self._unregularized_loss_batch, name="unregularized_loss")
        if self.__regularization_batch is None:
            self._regularization = tf.constant(0, name="regularization")
            self._loss = tf.identity(self._unregularized_loss, name="loss")
        else:
            self._regularization = tf.reduce_mean(regularization_lambda * self.__regularization_batch, name="regularization")
            self._loss = tf.add(self._unregularized_loss, self._regularization, name="loss")

        # self._loss = tf.reduce_mean(tf.square(self.output - target_batch), name="loss")
        if verbose:
            print("Shape of loss: {}".format(self._loss.shape))

        self._loss_min = tf.reduce_min(self._unregularized_loss_batch, name="loss_min")
        self._loss_max = tf.reduce_max(self._unregularized_loss_batch, name="loss_max")

        self._gradients = tf.gradients(self._loss, self.trainable_variables)
        self._gradients_and_variables = list(zip(self.gradients, self.trainable_variables))

        self._regularized_variables = list(set(tf.get_collection(self._regularized_variables_collection)))

    @property
    def gradients(self):
        return self._gradients

    @property
    def gradients_and_variables(self):
        return self._gradients_and_variables

    @property
    def regularized_variables(self):
        return self._regularized_variables

    @property
    def module_names(self):
        return self._module_names

    @property
    def modules(self):
        return self._modules

    @property
    def modules_with_names(self):
        return zip(self.module_names, self.modules)

    @property
    def loss_batch(self):
        return self._loss_batch

    @property
    def unweighted_loss(self):
        return self._unweighted_loss

    @property
    def unregularized_loss(self):
        return self._unregularized_loss

    @property
    def regularization(self):
        return self._regularization

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
        self._add_trainable_variables(models[0].trainable_variables)
        self._add_global_variables(models[0].global_variables)
        self._add_local_variables(models[0].local_variables)
        for name in models[0].summaries.keys():
            concat_tensors = tf.concat([model.summaries[name] for model in models], axis=0)
            self._add_summary(name, concat_tensors)

        concat_output = tf.concat([model.output for model in models], axis=0)
        self._set_output(concat_output)

        sum_unweighted_loss = tf.add_n([model.unweighted_loss for model in models])
        mean_unweighted_loss = tf.multiply(sum_unweighted_loss, 1.0 / len(models))
        self._unweighted_loss = mean_unweighted_loss

        sum_unregularized_loss = tf.add_n([model.unregularized_loss for model in models])
        mean_unregularized_loss = tf.multiply(sum_unregularized_loss, 1.0 / len(models))
        self._unregularized_loss = mean_unregularized_loss

        sum_regularization = tf.add_n([model.regularization for model in models])
        mean_regularization = tf.multiply(sum_regularization, 1.0 / len(models))
        self._regularization = mean_regularization

        sum_loss = tf.add_n([model.loss for model in models])
        mean_loss = tf.multiply(sum_loss, 1.0 / len(models))
        self._loss = mean_loss

        loss_min_concat = tf.stack([model.loss_min for model in models], axis=0)
        self._loss_min = tf.reduce_min(loss_min_concat, axis=0)

        loss_max_concat = tf.stack([model.loss_max for model in models], axis=0)
        self._loss_max = tf.reduce_max(loss_max_concat, axis=0)

        self._gradients = self._mean_gradients([model.gradients for model in models], verbose)
        self._gradients_and_variables = list(zip(self.gradients, self.trainable_variables))


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
    def gradients_and_variables(self):
        return self._gradients_and_variables

    @property
    def regularized_variables(self):
        return self._models[0]._regularized_variables

    @property
    def modules(self):
        return self._models[0].modules

    @property
    def modules_with_names(self):
        return self._models[0].modules_with_names

    @property
    def unweighted_loss(self):
        return self._unweighted_loss

    @property
    def unregularized_loss(self):
        return self._unregularized_loss

    @property
    def regularization(self):
        return self._regularization

    @property
    def loss(self):
        return self._loss

    @property
    def loss_min(self):
        return self._loss_min

    @property
    def loss_max(self):
        return self._loss_max
