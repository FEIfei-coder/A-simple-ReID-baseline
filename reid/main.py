from __future__ import absolute_import

import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import yaml
import os
import os.path as osp
import time
import math
import numpy as np

import Datasets
from Datasets.DataLoaders import *
import models
import loss_fuc
from utils.dataset import transform as T
from utils.dataset import preprocessor
from utils.dataset import optimizer
from utils.dataset.samplers import RandomIdentitySampler
from utils.logging import *
from utils.serialization import save_checkpoint, load_checkpoint

from tester import *
from trainer import *

from IPython import embed



def main():
	# load the hyper-parameter
	with open('config.yml', encoding='utf-8') as f:
		CONFIG_DICT = yaml.safe_load(f)  # CONFIG_DICT is a dict that involves train_param, test_param and save_dir
	TRAIN_PARAM = CONFIG_DICT['train']
	TEST_PARAM = CONFIG_DICT['test']
	SAVA_DIR = CONFIG_DICT['save_path']
	os.environ['CUDA_VISIBLE_DEVICES'] = TRAIN_PARAM['gpu_device']
	torch.manual_seed(TRAIN_PARAM['seed'])

	if not TRAIN_PARAM['evaluate']:
		sys.stdout = Logging(osp.join(SAVA_DIR['log_dir'], 'log_train.txt'))
	else:
		sys.stdout = Logging(osp.join(SAVA_DIR['log_dir'], 'log_test.txt'))
	print("==========\nArgs:{}\n==========".format(TRAIN_PARAM))

	# GPU use Y/N
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		print("Currently using GPU {}".format([TRAIN_PARAM['gpu_device']]))
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(TRAIN_PARAM['seed'])
	else:
		use_cpu = True

	print("Initializing dataset {}".format(TRAIN_PARAM['dataset']))
	# load data
	dataset = Datasets.init_dataset(name=TRAIN_PARAM['dataset'], root=TRAIN_PARAM['root'])

	pin_memory = True if use_gpu else False

	# define the tranform method
	train_transform = T.Compose([
		T.RandomSizedRectCrop(width=TRAIN_PARAM['width'], height=TRAIN_PARAM['height']),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	test_transform = T.Compose([
		T.RectScale(width=TRAIN_PARAM['width'], height=TRAIN_PARAM['height']),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


	train_loader = DataLoader(
		dataset=ImageDataset(dataset.train, transform=train_transform),
		sampler=RandomIdentitySampler(dataset.train, num_instances=TRAIN_PARAM['num_instances']),
		batch_size=TRAIN_PARAM['train_batch'],
		num_workers=TRAIN_PARAM['workers']	,
		pin_memory=pin_memory,
		drop_last=True
	)

	query_loader = DataLoader(
		dataset=ImageDataset(dataset=dataset.query, transform=test_transform),
		batch_size=TEST_PARAM['test_batch'],
		shuffle=False,
		num_workers=TEST_PARAM['test_workers']	,
		pin_memory=pin_memory,
		drop_last=False
	)

	gallery_loader = DataLoader(
		dataset=ImageDataset(dataset=dataset.gallery, transform=test_transform),
		batch_size=TEST_PARAM['test_batch'],
		shuffle=False,
		num_workers=TEST_PARAM['test_workers']	,
		pin_memory=pin_memory,
		drop_last=False
	)

	# load model
	print("Initializing model: {}".format(TRAIN_PARAM['arch']))
	model = models.init_model(
		name=TRAIN_PARAM['arch'],
		num_classes=dataset.num_train_pids,
		loss={'xent', 'htri'}
	)
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

	# load loss_fuc
	# we judge if the softmax / triHard is in the TRAIN_PARAM['losses'], or else setting None
	criterion_xent = loss_fuc.init_losses(
		name='softmax',
		num_classes=dataset.num_train_pids,
		use_gpu=use_gpu
	) if 'softmax' in TRAIN_PARAM['losses'] else None

	criterion_trihard = loss_fuc.init_losses(
		name='trihard',
		margin=TRAIN_PARAM['margin']
	) if 'trihard' in TRAIN_PARAM['losses'] else None

	# load optim
	optim = optimizer.init_optim(
		optim=TRAIN_PARAM['optim'],
		params=model.parameters(),
		lr=TRAIN_PARAM['lr'],
		weight_decay=TRAIN_PARAM['weight_decay']
	)
	if TRAIN_PARAM['step_size'] > 0:
		scheduler = lr_scheduler.StepLR(
			optimizer=optim,
			step_size=TRAIN_PARAM['step_size'],
			gamma=TRAIN_PARAM['gamma']
		)
	start_epoch = TRAIN_PARAM['start_epoch']

	# resume or not
	if TRAIN_PARAM['resume']:
		checkpoint = load_checkpoint(TRAIN_PARAM['resume'])
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']
		best_top1 = checkpoint['best_top1']
		print("=> Start epoch {}  best top1 {:.1%}"
			  .format(start_epoch, best_top1))

	if use_gpu:
		model = nn.DataParallel(model).cuda()
		criterion_trihard.cuda()
		criterion_xent.cuda()

	# test or not
	if TRAIN_PARAM['evaluate']:
		print("Evaluate only")
		# test(model, query_loader, gallery_loader, use_gpu)
		return

	start_time = time.time()
	train_time = 0
	best_rank1 = -np.inf
	best_epoch = 0
	print("==> Start training")

	# instance the class Trainer
	trainer = Trainer(
		model=model, criterion_xent=criterion_xent,
		criterion_trihard=criterion_trihard,
		eval=TRAIN_PARAM['triHard_only']
	)

	# start train
	for epoch in range(TRAIN_PARAM['start_epoch'], TRAIN_PARAM['max_epoch']):
		start_train_time =time.time()
		trainer.train(
			epoch=epoch,
			optimizer=optim,
			data_loader=train_loader,
			use_gpu=use_gpu,
			print_freq=TRAIN_PARAM['print_freq']
		)
		train_time += round(time.time() - start_train_time)

	#
	if (epoch + 1) > TEST_PARAM['start_eval'] \
			and TEST_PARAM['eval_step'] > 0 \
			and (epoch + 1) % TEST_PARAM['eval_step'] == 0 or (
			epoch + 1) == TRAIN_PARAM['max_epoch']:
		print("==> Test")
		rank1 = test(model, query_loader, gallery_loader, use_gpu)
		is_best = rank1 > best_rank1
		if is_best:
			best_rank1 = rank1
			best_epoch = epoch + 1

		if use_gpu:
			state_dict = model.module.state_dict()
		else:
			state_dict = model.state_dict()
		save_checkpoint({
			'state_dict': state_dict,
			'rank1': rank1,
			'epoch': epoch,
		}, is_best, osp.join(TRAIN_PARAM['checkpoint_dir'], 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

	print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	train_time = str(datetime.timedelta(seconds=train_time))
	print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

if __name__ == '__main__':
	main()

