from __future__ import absolute_import

import torch
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image

from IPython import embed

# read img
def read_img(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img_path, pid, camid = self.dataset[index]
		img = read_img(img_path)
		if self.transform is not None:
			img = self.transform(img)
		return img, pid, camid


# if __name__ == '__main__':
# 	market1501 = Market1501(root='D:\DesktopFile\Tasks\CVPaper\ReID_code\market1501')
# 	imgdataset = ImageDataset(market1501.train)
# 	loader = DataLoader(
# 		imgdataset,
# 		batch_size=4,
# 		shuffle=False,
# 		num_workers=4,
# 		drop_last=True
# 	)
#

