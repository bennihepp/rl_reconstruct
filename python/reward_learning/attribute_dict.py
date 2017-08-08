from __future__ import print_function


class AttributeDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))

    @staticmethod
    def convert_deep(dict_tree):
        attr_dict = AttributeDict(dict_tree)
        for name in attr_dict.keys():
            value = attr_dict[name]
            if type(value) == dict:
                attr_dict[name] = AttributeDict.convert_deep(value)
        return attr_dict
