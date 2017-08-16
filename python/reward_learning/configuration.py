from __future__ import print_function


def update_config_from_other(config, other):
    for key in other:
        if key in config:
            if isinstance(other[key], dict):
                assert(isinstance(config[key], dict))
                update_config_from_other(config[key], other[key])
            else:
                config[key] = other[key]
        else:
            config[key] = other[key]


def get_config_from_cmdline(args, topics, topic_mappings=None):
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
    return config


def update_config_from_cmdline(config, args, topic_mappings=None):
    if topic_mappings is None:
        topic_mappings = {}
    for topic in config:
        cmdline_topic = topic_mappings.get(topic, topic)
        for attr in dir(args):
            prefix = cmdline_topic + "."
            if attr.startswith(prefix):
                name = attr[len(prefix):]
                config[topic][name] = getattr(args, attr)
    return config
