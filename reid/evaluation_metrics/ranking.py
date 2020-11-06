from __future__ import absolute_import

import numpy as np
from numpy import ndarray
from collections import defaultdict

import torch
import torch.nn

from IPython import embed
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
	mask = np.zeros(num, dtype=np.bool)
	for _, indices in ids_dict.items():
		i = np.random.choice(indices)
		mask[i] = True
	return mask


def cmc(distmat: ndarray, q_pids: ndarray=None,
		g_pids: ndarray=None, q_camids: ndarray=None,
		g_camids: ndarray=None, max_rank=None,
		separate_camera_id=False,
		single_gallery_shot=False,
		first_match_break=False):
	'''
	:Key:
		for each query identity, its gallery images from the same camera view are discarded.

	:param distmat: distance matrix between query feature and gallery feature
	:param q_pids: q-id
	:param g_pids: g-id
	:param q_camids: cam-id from q
	:param g_camids: cam-id from g
	:param max_rank: [1, 5, 10, 20]
	:param separate_camera_id: market1501=Ture
	:param single_gallery: multi / single
	:param first_match_break:
	:return: cmc
	'''
	m, n = distmat.shape # m: the num of q ids,  n: the num of g ids

	# Fill up default values
	if q_pids is None:
		q_pids = np.arange(m)
	if g_pids is None:
		g_pids = np.arange(n)
	if q_camids is None:
		q_camids = np.zeros(m).astype(np.int32)
	if g_camids is None:
		g_camids = np.ones(n).astype(np.int32)

	# Ensure numpy array
	query_ids = np.asarray(q_pids)
	gallery_ids = np.asarray(g_pids)
	query_cams = np.asarray(q_camids)
	gallery_cams = np.asarray(g_camids)

	# sort the distance from each q to g
	indices = distmat.argsort(axis=1)
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

	# compute cmc
	ret = np.zeros(max_rank)
	num_valid_query = 0
	for i in range(m):
		# Filter out the same id and same camera
		valid = ((g_pids[indices[i]] != q_pids[i]) |
				 (g_pids[indices[i]] != q_pids[i]))
		if separate_camera_id:
			# Filter out the same camera
			valid &= (g_camids[indices[i]] != q_camids[i])
		if not np.any(matches[i, valid]):
			continue
		if single_gallery_shot:
			repeat = 10
			gids = g_pids[indices[i][valid]]
			inds = np.where(valid)[0]
			ids_dict = defaultdict(list)
			for j, x in zip(inds, gids):
				ids_dict[x].append(j)
		else:
			repeat = 1
		for _ in range(repeat):
			if single_gallery_shot:
				# Randomly choose one instance for each id
				sampled = (valid & _unique_sample(ids_dict, len(valid)))
				index = np.nonzero(matches[i, sampled])[0]
			else:
				index = np.nonzero(matches[i, valid])[0]
			delta = 1. / (len(index) * repeat)
			for j, k in enumerate(index):
				if k - j >= max_rank: break
				if first_match_break:
					ret[k - j] += 1
					break
				ret[k - j] += delta

		num_valid_query += 1
		if num_valid_query == 0:
			raise RuntimeError("No valid query")
	embed()
	return ret.cumsum() / num_valid_query



def mean_ap(distmat, query_ids=None,
			gallery_ids=None,query_cams=None,
			gallery_cams=None):
	m, n = distmat.shape
    # Fill up default values
	if query_ids is None:
		query_ids = np.arange(m)
	if gallery_ids is None:
		gallery_ids = np.arange(n)
	if query_cams is None:
		query_cams = np.zeros(m).astype(np.int32)
	if gallery_cams is None:
		gallery_cams = np.ones(n).astype(np.int32)
	# Ensure numpy array
	query_ids = np.asarray(query_ids)
	gallery_ids = np.asarray(gallery_ids)
	query_cams = np.asarray(query_cams)
	gallery_cams = np.asarray(gallery_cams)

	# Sort and find correct matches
	indices = np.argsort(distmat, axis=1)
	matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
	# Compute AP for each query
	aps = []
	for i in range(m):
		# Filter out the same id and same camera
		valid = ((gallery_ids[indices[i]] != query_ids[i]) |
				 (gallery_cams[indices[i]] != query_cams[i]))
		y_true = matches[i, valid]
		y_score = -distmat[i][indices[i]][valid]
		if not np.any(y_true): continue
		aps.append(average_precision_score(y_true, y_score))
	if len(aps) == 0:
		raise RuntimeError("No valid query")
	return np.mean(aps)



if __name__ == '__main__':
	a = np.random.random_integers(low=1, high=10, size=(4, 10))
	p = cmc(a)
	print(p)
