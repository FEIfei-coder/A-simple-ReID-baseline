from __future__ import absolute_import

import time

import numpy as np
import torch

from utils.meter import AverageMeter


class BaseTester(object):
	def __init__(self, model, name, test_batch):
		'''
		:param model: the model has been trained
		:param name: the dist function name.
					 e.g cosine dist, euclidean dist, Jaccard dist...
		'''
		self.model = model
		self.name = name
		self.test_batch = test_batch


	def test(self, query_loader, gallery_loader, use_gpu, ranks=None):
		'''
		:param model: trained model
		:param query_loader: query-larder
		:param gallery_loader: gallery-loader
		:param use_gpu: given from main.py
		:param ranks: the ranks we need
		:return: CMC / mAP
		'''
		self.model.eval()

		batch_time = AverageMeter()

		with torch.no_grad():
			# get the feature / pids / camids from the data_loader
			qf, q_pids, q_camids, batch_time = self._extract_feature(query_loader, use_gpu, batch_time)
			gf, g_pids, g_camids, batch_time = self._extract_feature(gallery_loader, use_gpu, batch_time)

		print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, self.test_batch))


	def _pair_dist(self, mat1, mat2):
		'''
		:param mat1: matrix 1 : tensor
		:param mat2: matrix 2 : tensor
		:return: dist between mat1 & mat2
		'''
		m, n = mat1.size(0), mat2.size(0)


	def _extract_feature(self, data_loader, use_gpu, batch_time, print_freq=1, metric=None):
		'''
		:param data_loader: query_loader / gallery_loader
		:param print_freq: default 1
		:param use_gpu: delivered from main.py
		:param batch_time:
		:param metric:
		:return: feature & pids & camids & batch_time
		'''
		raise NotImplementedError



class Tester(BaseTester):
	'''
	the test fuc we use
	'''
	def _extract_feature(self, data_loader, use_gpu, batch_time, print_freq=1, metric=None):
		features, pids, camids = [], [], []
		for batch_idx, (imgs, pids, camids) in enumerate(data_loader):
			if use_gpu:
				imgs = imgs.cuda()

			end = time.time()
			features = self.model(imgs)  # resnet50: f
			batch_time.update(time.time() - end)

			features = features.data.cpu()
			features.append(features)
			pids.extend(pids)
			camids.extend(camids)
		features = torch.cat(features, 0)
		pids = np.asarray(pids)
		camids = np.asarray(camids)

		print("Extracted features for {} set, obtained {}-by-{} matrix".
			  format(data_loader, features.size(0), features.size(1)))

		return features, pids, camids, batch_time




def extract_feature(model, data_loader, use_gpu, print_freq=1, metric=None):
	'''
	:param model: the model has been trained
	:param data_loader: query_loader / gallery_loader
	:param print_freq: default 1
	:param use_gpu: delivered from main.py
	:param metric:
	:return: feature & pids & camids
	'''
	model.eval()
	batch_time = AverageMeter()

	features, pids, camids = [], [], []
	for batch_idx, (imgs, pids, camids) in enumerate(data_loader):
		if use_gpu:
			imgs = imgs.cuda()

		end = time.time()
		features = model(imgs) # resnet50: f
		batch_time.update(time.time() - end)

		features = features.data.cpu()
		features.append(features)
		pids.extend(pids)
		camids.extend(camids)
	features = torch.cat(features, 0)
	pids = np.asarray(pids)
	camids = np.asarray(camids)

	print("Extracted features for {} set, obtained {}-by-{} matrix".
		  format(data_loader, features.size(0), features.size(1)))

	return features, pids, camids


def dist_pairwise():
	pass


def test(model, query_loader, gallery_loader, use_gpu, ranks=None):
	'''
	:param model: trained model
	:param query_loader: query-larder
	:param gallery_loader: gallery-loader
	:param use_gpu: given from main.py
	:param ranks: the ranks we need
	:return: CMC / mAP
	'''
	model.eval()

	# get the feature / pids / camids from the data_loader
	qf, q_pids, q_camids = extract_feature(model, query_loader, use_gpu)
	gf, g_pids, g_camids = extract_feature(model, gallery_loader, use_gpu)


