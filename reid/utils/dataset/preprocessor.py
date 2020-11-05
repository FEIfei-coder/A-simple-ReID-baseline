from __future__ import absolute_import

import os.path as osp
from PIL import Image

class Preprocessor(object):
	def __init__(self, dataset, transform=None, root=None):
		super(Preprocessor, self).__init__()
		self.dataset = dataset
		self.transform = transform
		self.root = root

	def __len__(self):
		return len(self.dataset)


	def __getitem__(self, indice):
		if isinstance(indice, (tuple, list)):
			return [self._get_single_item(index) for index in indice]
		if isinstance(indice, int):
			return self._get_single_item(indice)


	def _get_single_item(self, index):
		img_path, pid, camid = self.dataset[index]
		# if self.root is not None:
		# 	fpath = osp.join(self.root, img_path)
		img = Image.open(img_path, 'r').convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		return img, pid, camid
