from __future__ import absolute_import

from .DataManagers import *

__factory = {
	'market1501': Market1501
}

def get_factory():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in get_factory():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
