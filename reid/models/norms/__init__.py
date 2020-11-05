from __future__ import absolute_import



__factory = {
}

def get_name():
	return __factory.keys()


def init_attention(name, *args, **kwargs):
	if name not in get_name():
		raise KeyError('Unknown norm module:{}'.format(name))
	return __factory[name](*args, **kwargs)
