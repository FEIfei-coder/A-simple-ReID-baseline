from .triHard import *
from .softmaxCE import *
from .deepSupervision import *

__factory = {
	'softmax': CrossEntropyLabelSmooth,
	'trihard': TriHardLoss,
	'deepsupervise': DeepSupervision
}

def get_name():
	return __factory.keys()

def init_losses(name, *args, **kwargs):
	if name not in get_name():
		raise KeyError('unkown loss_fuc:{}'.format(name))
	return __factory[name](*args, **kwargs)
