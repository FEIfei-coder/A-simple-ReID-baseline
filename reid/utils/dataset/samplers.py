from __future__ import absolute_import

import torch
from torch.utils.data import Sampler, SequentialSampler, WeightedRandomSampler, BatchSampler

from collections import defaultdict
import numpy as np


class RandomIdentitySampler(Sampler):
	'''
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
	'''
	def __init__(self, dataset, num_instances=4):
		'''
		:param dataset: the dataset to sample from
		:param num_instances: each batch how many ids we sample
		'''
		self.dataset = dataset
		self.num_instances = num_instances
		self.index_dic = defaultdict(list) # we want a dict which can be operated
		for index, (_, pid, _) in enumerate(dataset):
			self.index_dic[pid].append(index)
		self.pids = list(self.index_dic.keys())
		self.num_pids = len(self.pids)


	def __iter__(self):
		'''
		:return: the list within sampled pid data in num_instances
		'''
		indices = torch.randperm(self.num_pids) # derange the index of the pids
		ret = []
		for i in indices:
			pid = self.pids[i] # get the deranged pid
			t = self.index_dic[pid] # get the data(class:list) in each pid
			if len(t) >= self.num_instances:
				# if the num_data>=num_instances, we don't replace any other data and sample from t in num_instances
				t = np.random.choice(t, size=self.num_instances, replace=False)
			else:
				# # if the num_data<num_instances, we get all of the data and replace (num_instances-len(t)) data
				t = np.random.choice(t, size=self.num_instances, replace=True)
			ret.extend(t)
		return iter(ret)


	def __len__(self):
		'''
		:return: total num of the sample
		'''
		return self.num_instances * self.num_pids


