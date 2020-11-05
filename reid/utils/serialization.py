from __future__ import absolute_import

import json
import os.path as osp
import shutil

import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_json(fpath):
	if osp.exists(fpath):
		with open(fpath, 'r') as f:
			obj = json.load(f)
		return obj
	else:
		raise IOError('The file path is error')


def write_json(obj, fpath):
	'''
	:param obj: the object we wanna write in
	:param fpath: the file we wanna write
	'''
	mkdir_if_missing(osp.dirname(fpath))
	with open(fpath, 'w') as f:
		json.dump(obj, f, indent=4, separators=(',', ':'))


def save_checkpoint(state, is_best, fpath='checkpoint.pth.taz'):
	'''
	:param state: model.state_dict()  file.pth
	:param is_best: is the model
	:param fpath: saving dir
	'''
	mkdir_if_missing(osp.dirname(fpath))
	torch.save(state, fpath)
	if is_best:
		shutil.copy(fpath, osp.join(osp.join(osp.dirname(fpath), 'model_best.pth.tar')))


def load_checkpoint(fpath):
	if osp.isfile(fpath):
		checkpoint = torch.load(fpath)
		print("=> Loaded checkpoint '{}'".format(fpath))
		return checkpoint
	else:
		raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
	'''
	:param state_dict: state-dict of the model
	:param model: the model we train
	:param strip:
	'''
	tgt_state = model.state_dict()
	copied_names = set()
	for name, param in state_dict.items():
		if strip is not None and name.startswith(strip):
			name = name[len(strip):]
		if name not in tgt_state:
			continue
		if isinstance(param, Parameter):
			param = param.data
		if param.size() != tgt_state[name].size():
			print('mismatch:', name, param.size(), tgt_state[name].size())
			continue
		tgt_state[name].copy_(param)
		copied_names.add(name)

	missing = set(tgt_state.keys()) - copied_names
	if len(missing) > 0:
		print("missing keys in state_dict:", missing)

	return model


