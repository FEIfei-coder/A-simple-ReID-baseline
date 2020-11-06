from .mINP import *
from .ranking import *
from .rerank import *

__factory = {

}

def get_factory():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in get_factory():
        raise KeyError("Unknown evaluate method: {}".format(name))
    return __factory[name](*args, **kwargs)
