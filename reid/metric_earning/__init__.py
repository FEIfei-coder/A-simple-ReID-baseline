from .cosine import *
from .euclidean import *
from .jaccard import *
from .norm import *

__factory = {
	'norm': norm_,
	'euclidean': euclidean_,
	'cosine': cosine_,
	'jaccard': jaccard_
}

def get_name():
	return __factory.keys()

def init_losses(name, *args, **kwargs):
	if name not in get_name():
		raise KeyError('unkown dist-func:{}'.format(name))
	return __factory[name](*args, **kwargs)
