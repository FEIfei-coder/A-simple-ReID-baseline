from __future__ import absolute_import

from .cam import *
from .sam import *
from .bam import *
from .cbam import *
from .sa import *


__factory = {
	'seblock': ChannelAttention_,
	'channel': ChannelAttention,
	'spatial': SpatialAttention_,
	'sam_cbam': SpatialAttention,
	'bam': BAM_Block,
	'cbam': CBAM_Block,
	'non_local': NonLocal,
	'GCblock': GCBlock,
	'CCblock': CCBlock
}

def get_name():
	return __factory.keys()


def init_attention(name, *args, **kwargs):
	if name not in get_name():
		raise KeyError('Unknown attention mechanism:{}'.format(name))
	return __factory[name](*args, **kwargs)

