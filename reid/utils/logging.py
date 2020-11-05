from __future__ import absolute_import

import os
import sys

from .osutils import mkdir_if_missing

class Logging(object):
	'''
	this module provides access to handle the log
	for this class, we set 5 methods

	the core method of the class is the self.write(), self.flush() and self.close()
	'''
	def __init__(self, fpath=None):
		self.console = sys.stdout # redirect I/O
		self.file = None
		if fpath is not None:
			mkdir_if_missing(os.path.dirname(fpath))
			self.file = open(fpath, 'w')

	def __del__(self):
		self.close()


	def __enter__(self):
		pass

	def __exit__(self, *args):
		self.close()

	def write(self, msg):
		'''
		:param msg: the message we wanna write in the log
		'''
		self.console.write(msg)
		if self.file is not None:
			self.file.write(msg)

	def flush(self):
		self.console.flush()
		if self.file is not None:
			self.file.flush()
			os.fsync(self.file.fileno())

	def close(self):
		self.console.close()
		if self.file is not None:
			self.file.close()


