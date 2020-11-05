from __future__ import absolute_import

import os
import errno

def mkdir_if_missing(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

# if __name__ == '__main__':
# 	mkdir_if_missing('D:\\DesktopFile\\Tasks\\CVPaper\\ReID_code\\MyBaseline\\reid\\log')



