from __future__ import absolute_import

from .triHard import *
from .softmaxCE import *
from .deepSupervision import *
from .metricloss import *


__factory = {
	'softmax': CrossEntropyLabelSmooth,
	'trihard': TriHardLoss,
	'deepsupervise': DeepSupervision,
	'arcface': arcface_,
	'norm': normface_,
	'am-softmax': am_softmax_,
	'a-softmax': a_softmax_,
	'softtrip': soft_triplet_,
	'contrastive': contrastive_loss_,
	'n-pair': n_pairs_loss_,
	'tuple': tuplet_margin_loss_,
	'lift-loss': lifted_structure_loss_,
	'circle-loss': circle_loss_
}

def get_name():
	return __factory.keys()

def init_losses(name, *args, **kwargs):
	if name not in get_name():
		raise KeyError('unkown loss_fuc:{}'.format(name))
	return __factory[name](*args, **kwargs)
