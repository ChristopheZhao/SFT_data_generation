import json

class cfg_dict(object):

    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

    def getitem(self, key):
        return self.__dict__.get(key)
    


def load_config(dict1):

    return json.loads(json.dumps(dict1), object_hook=cfg_dict)