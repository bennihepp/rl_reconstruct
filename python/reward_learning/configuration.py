from __future__ import print_function


def update_config_from_other(config, other, verbose=False):
    for key in other:
        if key in config:
            if isinstance(other[key], dict):
                assert(isinstance(config[key], dict))
                if verbose:
                    print("Recursing into config namespace: {}".format(key))
                update_config_from_other(config[key], other[key])
            elif other[key] is not None:
                config[key] = other[key]
                if verbose:
                    print("Updating config item: {} <- {}".format(key, other[key]))
        else:
            config[key] = other[key]
            if verbose:
                print("Adding new config item: {} <- {}".format(key, other[key]))


def get_config_from_cmdline(args, topics, topic_mappings=None, verbose=False):
    config = {}
    if topic_mappings is None:
        topic_mappings = {}
    for topic in topics:
        if topic not in config:
            config[topic] = {}
        cmdline_topic = topic_mappings.get(topic, topic)
        for attr in dir(args):
            prefix = cmdline_topic + "."
            if attr.startswith(prefix):
                name = attr[len(prefix):]
                config[topic][name] = getattr(args, attr)
                if verbose:
                    print("Config item: {}.{}".format(topic, name))
    return config


def update_config_from_cmdline(config, args, topic_mappings=None, verbose=False):
    if topic_mappings is None:
        topic_mappings = {}
    for topic in config:
        cmdline_topic = topic_mappings.get(topic, topic)
        for attr in dir(args):
            prefix = cmdline_topic + "."
            if attr.startswith(prefix):
                name = attr[len(prefix):]
                config[topic][name] = getattr(args, attr)
                if verbose:
                    print("Updating config item: {}.{} <- {}".format(topic, name, config[topic][name]))
    return config
