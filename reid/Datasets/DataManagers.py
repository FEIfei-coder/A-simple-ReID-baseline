from __future__ import absolute_import

import os.path as osp
import glob
import re


class Market1501(object):

	market1501 = 'Market-1501-v15.09.15'

	def __init__(self, root, *args, **kwargs):
		self.dataset_dir = osp.join(root, self.market1501)
		self.dataset_train = osp.join(self.dataset_dir, 'bounding_box_train')
		self.dataset_test = osp.join(self.dataset_dir, 'bounding_box_test')
		self.query_dataset = osp.join(self.dataset_dir, 'query')

		self._check_dir()

		train, num_train_pids, num_train_imgs = self._process_dataset(self.dataset_train)
		gallery, num_gallery_pids, num_gallery_imgs = self._process_dataset(self.dataset_test)
		query, num_query_pids, num_query_imgs = self._process_dataset(self.query_dataset)

		num_total_pids = num_train_pids + num_query_pids
		num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

		print("=> Market1501 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  subset   | # ids | # images")
		print("  ------------------------------")
		print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
		print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
		print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
		print("  ------------------------------")
		print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
		print("  ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

	def _check_dir(self):
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.dataset_train):
			raise RuntimeError("'{}' is not available".format(self.dataset_train))
		if not osp.exists(self.dataset_test):
			raise RuntimeError("'{}' is not available".format(self.dataset_test))
		if not osp.exists(self.query_dataset):
			raise RuntimeError("'{}' is not available".format(self.query_dataset))


	def _process_dataset(self, dir_path, relabel=True):
		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		pattern = re.compile(r'([-\d]+)_c(\d)')

		pid_container = set()
		for img_path in img_paths:
			pid, _ = map(int, pattern.search(img_path).groups())
			if pid == -1: continue  # junk images are just ignored
			pid_container.add(pid)
		pid2label = {pid: label for label, pid in enumerate(pid_container)}

		dataset = []
		for img_path in img_paths:
			pid, camid = map(int, pattern.search(img_path).groups())
			if pid == -1: continue  # junk images are just ignored
			assert 0 <= pid <= 1501  # pid == 0 means background
			assert 1 <= camid <= 6
			camid -= 1  # index starts from 0
			if relabel: pid = pid2label[pid]
			dataset.append((img_path, pid, camid))

		num_pids = len(pid_container)
		num_imgs = len(dataset)

		return dataset, num_pids, num_imgs



