from __future__ import absolute_import

import time
import datetime

from utils.meter import AverageMeter
from loss_fuc import DeepSupervision


'''
Firstly, we define the BaseTrainer class
'''
class BaseTrainer(object):
	'''
	this is a basic Trainer
	'''
	def __init__(self, model, criterion_xent, criterion_trihard, eval):
		'''
		:param model: the model we've defined in main.py
		:param criterion: loss_fuc fuction loader from main.py
		'''
		super(BaseTrainer, self).__init__()
		self.model = model
		self.criterion_xent = criterion_xent
		self.criterion_trihard = criterion_trihard
		self.eval = eval


	def train(self, epoch, data_loader, optimizer, print_freq, use_gpu):
		'''
		:param epoch: train epoch
		:param data_loader: train_loader variable
		:param optimizer: optim we define in the main.py
		:param print_freq: in config.yml
		:param use_gpu: get from the torch.cuda.is_availible
		:param eval: in config.yml
		'''
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		# precisions = AverageMeter()

		end = time.time()
		for indx, inputs in enumerate(data_loader):
			'''
			because different dataset should be handled differently, so we set the function 
			self._parse_data() to get data for different dataset
			
			self._parse_data() ==
			# if use_gpu:
			#  	imgs, pids = imgs.cuda(), pids.cuda()
			'''
			imgs, pids = self._parse_data(inputs=inputs, use_gpu=use_gpu)

			data_time.update(time.time() - end)
			loss = self._forward(model=self.model, imgs=imgs, pids=pids, eval=self.eval)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), pids.size(0))

			if (indx + 1) % print_freq == 0:
				print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
					epoch + 1, indx + 1, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses))


	def _parse_data(self, inputs, use_gpu, *args, **kwargs):
		'''
		:Desribe: This method is defined for parse the element of the dataloader.
				  For implementing this method, you should overwrite it (likes abstract method)

		:param inputs: elements in dataloader
		:param use_gpu: ...
		:param args: ...
		:param kwargs: ...
		:return: imgs & pids
		'''
		raise NotImplementedError


	def _forward(self, model, imgs, pids, eval, *args, **kwargs):
		raise NotImplementedError


'''
then, we write the class -- Trainer
'''
class Trainer(BaseTrainer):
	'''
	For Baseline training
	'''
	def _parse_data(self, inputs, use_gpu, *args, **kwargs):
		imgs, pids, _ = inputs
		if use_gpu:
			imgs, pids = imgs.cuda(), pids.cuda()
		return imgs, pids


	def _forward(self, model, imgs, pids, eval, *args, **kwargs):
		outputs, features = self.model(imgs)

		if eval:
			if isinstance(features, tuple):
				loss = DeepSupervision(self.criterion_trihard, features, pids)
			else:
				loss = self.criterion_trihard(features, pids)
		else:
			if isinstance(outputs, tuple):
				xent_loss = DeepSupervision(self.criterion_xent, outputs, pids)
			else:
				xent_loss = self.criterion_xent(outputs, pids)

			if isinstance(features, tuple):
				htri_loss = DeepSupervision(self.criterion_trihard, features, pids)
			else:
				htri_loss = self.criterion_trihard(features, pids)

			loss = xent_loss + htri_loss

		return loss


class CrossModalityTrainer(BaseTrainer):
	'''
	For cross modality training
	'''

	def _parse_data(self, inputs, use_gpu, *args, **kwargs):
		pass

	def _forward(self, model, imgs, pids, eval, *args, **kwargs):
		pass


