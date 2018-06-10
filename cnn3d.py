#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import os, sys,subprocess
import random, time
import inspect
import copy
# mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# sys.path.append(mypydir)
# sys.path.append(mypydir+"/mytools")
import collections
import math
import random
import zipfile
import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
# from namehostip import get_my_ip
# from hostip import ip2tarekc,tarekc2ip
# from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
# from util import read_lines_as_list,read_lines_as_dic
# configfile = "conf.txt"
# taskfile = "task_list.txt"

iprint = 1

SCOPE_CONV = "conv%d"
SCOPE_NORM = "norm%d"
SCOPE_DROP = "drop%d"
SCOPE_POOL = "pool%d"
scope_cnt = 0
def scnt(inc=0, reset=False):
	global scope_cnt
	if reset:
		scope_cnt=0
	scope_cnt+=inc
	return scope_cnt

layers = tf.contrib.layers 
activation_layer = tf.nn.elu

def batch_norm_layer(inputs, phase_train, scope=None): # NHWC or NDHWC.!
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)

def cnn3d_pool(inputs, param_dict={}, train=True, name='cnn3d_pool'): # out (None, ind, 100)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	convNum6 =param_dict["convNum6"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]
	OUT_DIM = param_dict["OUT_DIM"]
	if train:
		reusing = False
	else:
		reusing = True
	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None,d, 25, 750) 
			pass
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		inputs = tf.expand_dims(inputs, axis=4)# (None, d, 25,750,1) 
		
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())
		padding_cv = "VALID"
		padding_pool = "SAME"
		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]
		pool_stride= [1,1,1,pool_size,1]

		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 1, 7],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 25, 372, 64)


		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 1, 5],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 25, 184, 64)


		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, 1, 4],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 25, 91, 64)


		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, conv_shape[2], 1],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 1, 91, 64)


		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 1, 4],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 1, 44, 64)


		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]

		while conv_shape[3]>21:
			conv = layers.convolution2d(conv, convNum6, kernel_size=[1, 1, 3],
							stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
			conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
			conv = activation_layer(conv)
			conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
			conv_shape = conv.get_shape().as_list()
			conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
				noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
			print(scnt(),conv_shape)
			# (None, d, 1, 21, 64)


		conv = layers.convolution2d(conv, convNum6, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		# conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
		# 	noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		# (None, d, 1, 8, 64)

		conv = tf.reshape(conv, [conv_shape[0], conv_shape[1], conv_shape[2]*conv_shape[3]*conv_shape[4]])
		conv_shape = conv.get_shape().as_list()
		print(scnt(1),conv_shape)
		#(None, d, 500)

		conv = layers.fully_connected(conv, OUT_DIM, activation_fn=activation_layer, scope='output')
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0],1, conv_shape[2]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)

		return conv

def cnn3d_pool_split(inputs, param_dict={}, train=True, name='cnn3d_pool_split'): # out (None, ensenble, 100)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	convNum6 =param_dict["convNum6"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]
	OUT_DIM = param_dict["OUT_DIM"]
	if train:
		reusing = False
	else:
		reusing = True
	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None, 1, 25, 750) 
			pass
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		inputs = tf.expand_dims(inputs, axis=4)# (None, 1, 25,750,1) 
		
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())
		padding_cv = "VALID"
		padding_pool = "SAME"
		pool_size = 3
		pool_ksize = [1,1,1,pool_size,1]
		pool_stride = [1,1,1,1,1]

		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 1, 6],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 1, 25, 750, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		num_split = conv_shape[3]
		splits = tf.split(conv,num_or_size_splits=num_split,axis=3) #list: [(None, 1, 25, 1, 64),...]
		ensenble = []
		minlen = 1e10
		for i in range(pool_size):
			slen = len(range(i,num_split,pool_size))
			if slen<minlen:
				minlen=slen
		for i in range(pool_size):
			indices = range(i,num_split,pool_size)
			gather=[]
			for j in range(minlen):
				gather.append(splits[indices[j]])
			concat=tf.concat(gather, axis=3)
			if iprint>=2 and i==0: print(scnt(),"concat",concat.get_shape().as_list())
			ensenble.append(concat)
		conv = tf.concat(ensenble, axis=1)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),"ensenble",conv_shape)
		# (None, 3, 25, 250, 64)


		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 1, 5],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 3, 25, 250, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		num_split = conv_shape[3]
		splits = tf.split(conv,num_or_size_splits=num_split,axis=3) #list: [(None, 1, 25, 1, 64),...]
		ensenble = []
		minlen = 1e10
		for i in range(pool_size):
			slen = len(range(i,num_split,pool_size))
			if slen<minlen:
				minlen=slen
		for i in range(pool_size):
			indices = range(i,num_split,pool_size)
			gather=[]
			for j in range(minlen):
				gather.append(splits[indices[j]])
			concat=tf.concat(gather, axis=3)
			if iprint>=2 and i==0: print(scnt(),"concat",concat.get_shape().as_list())
			ensenble.append(concat)
		conv = tf.concat(ensenble, axis=1)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),"ensenble",conv_shape)
		# (None, 9, 25, 112, 64)


		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, conv_shape[2], 1],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 9, 1, 112, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, 1, 4],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 9, 1, 112, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		num_split = conv_shape[3]
		splits = tf.split(conv,num_or_size_splits=num_split,axis=3) #list: [(None, 1, 1, 1, 64),...]
		ensenble = []
		minlen = 1e10
		for i in range(pool_size):
			slen = len(range(i,num_split,pool_size))
			if slen<minlen:
				minlen=slen
		for i in range(pool_size):
			indices = range(i,num_split,pool_size)
			gather=[]
			for j in range(minlen):
				gather.append(splits[indices[j]])
			concat=tf.concat(gather, axis=3)
			if iprint>=2 and i==0: print(scnt(),"concat",concat.get_shape().as_list())
			ensenble.append(concat)
		conv = tf.concat(ensenble, axis=1)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),"ensenble",conv_shape)
		# (None, 27, 1, 37, 64)


		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]
		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 27, 1, 37, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		num_split = conv_shape[3]
		splits = tf.split(conv,num_or_size_splits=num_split,axis=3) #list: [(None, 1, 1, 1, 64),...]
		ensenble = []
		minlen = 1e10
		for i in range(pool_size):
			slen = len(range(i,num_split,pool_size))
			if slen<minlen:
				minlen=slen
		for i in range(pool_size):
			indices = range(i,num_split,pool_size)
			gather=[]
			for j in range(minlen):
				gather.append(splits[indices[j]])
			concat=tf.concat(gather, axis=3)
			if iprint>=2 and i==0: print(scnt(),"concat",concat.get_shape().as_list())
			ensenble.append(concat)
		conv = tf.concat(ensenble, axis=1)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),"ensenble",conv_shape)
		# (None, 54, 1, 18, 64)


		conv = layers.convolution2d(conv, convNum6, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 54, 1, 18, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		# conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
		# 	noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		num_split = conv_shape[3]
		splits = tf.split(conv,num_or_size_splits=num_split,axis=3) #list: [(None, 1, 1, 1, 64),...]
		ensenble = []
		minlen = 1e10
		for i in range(pool_size):
			slen = len(range(i,num_split,pool_size))
			if slen<minlen:
				minlen=slen
		for i in range(pool_size):
			indices = range(i,num_split,pool_size)
			gather=[]
			for j in range(minlen):
				gather.append(splits[indices[j]])
			concat=tf.concat(gather, axis=3)
			if iprint>=2 and i==0: print(scnt(),"concat",concat.get_shape().as_list())
			ensenble.append(concat)
		conv = tf.concat(ensenble, axis=1)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),"ensenble",conv_shape)
		# (None, 108, 1, 8, 64)

		conv = tf.reshape(conv, [conv_shape[0], conv_shape[1], conv_shape[2]*conv_shape[3]*conv_shape[4]])
		conv_shape = conv.get_shape().as_list()
		print(scnt(1),conv_shape)
		#(None, 108, 500)

		conv = layers.fully_connected(conv, OUT_DIM, activation_fn=activation_layer, scope='output')
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0],1, conv_shape[2]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)

		return conv

def cnn3d_dilated(inputs, param_dict, train, name='cnn3d_dilated'): # did dropout at last layer!!! # (None, 4, 25, 125, 32)
	CVN1 =param_dict["CVN1"]
	CVN2 =param_dict["CVN2"]
	CVN3 =param_dict["CVN3"]
	CVN4 =param_dict["CVN4"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]

	if train:
		reusing = False
	else:
		reusing = True

	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		# (None, 4, 25, time) 
		if len(inputs.get_shape().as_list()) == 4:
			inputs = tf.expand_dims(inputs, axis=4)
		# [None, 4, 25, 60, 1]
		inputs_shape = inputs.get_shape().as_list()


		dilation = [1, 1, 1]
		f_shape = [1, 1, 3, inputs_shape[-1], CVN1]
		f=tf.get_variable('dilation_w%d'%scnt(1), shape = f_shape, dtype = tf.float32, initializer = tf.truncated_normal_initializer())
		sensor_conv2 = tf.nn.convolution(inputs, f, dilation_rate=dilation, padding='SAME', data_format='NDHWC',name=scope.name)
		# (None, 4, 25, 60, 32)
		sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope=SCOPE_NORM%scnt())
		sensor_conv2 = activation_layer(sensor_conv2)
		out_shape = sensor_conv2.get_shape().as_list()
		sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[out_shape[0], 1, 1, 1, out_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),out_shape)


		dilation = [1, 1, 2]
		f_shape = [1, 1, 3, CVN1, CVN2]
		f=tf.get_variable('dilation_w%d'%scnt(1), shape = f_shape, dtype = tf.float32,initializer = tf.truncated_normal_initializer())
		conv = tf.nn.convolution(sensor_conv2, f, dilation_rate=dilation, padding='SAME', data_format='NDHWC',name=scope.name)
		# (None, 4, 25, 60, 32)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		out_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[out_shape[0], 1, 1, 1, out_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),out_shape)

		return conv


def cnn3d(inputs, param_dict, train, name='cnn3d'): # no dropout at last layer!!! (None, 4, 50)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]

	if train:
		reusing = False
	else:
		reusing = True

	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)

		# inputs shape (None, 4, 25, 111) 
		print(name,inputs.get_shape().as_list())
		if len(inputs.get_shape().as_list()) == 4:
			inputs = tf.expand_dims(inputs, axis=4)

		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]
		pool_stride = [1,1,1,pool_size,1]
		padding_pool = "SAME"

		# [None, 4, 25, 60, 1]
		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 1, 6],
						stride=[1, 1, 1], padding='VALID', data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 4, 25, 30, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		padding = "SAME"
		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 1, 4],
						stride=[1, 1, 1], padding=padding, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 4, 25, 15, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, conv_shape[2], 1],
						stride=[1, 1, 1], padding='VALID', data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 4, 1, 15, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 4, 1, 7, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 1, 3],
						stride=[1, 1, 2], padding=padding, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 4, 1, 3, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt())
		conv_shape = conv.get_shape().as_list()
		print(scnt(),conv_shape)


		sensor_conv_out = tf.reshape(conv, [conv_shape[0], conv_shape[1], conv_shape[2]*conv_shape[3]*conv_shape[4]])
		sensor_conv_out_shape = sensor_conv_out.get_shape().as_list()
		print(scnt(),sensor_conv_out_shape)
		# (None, 4, 50)

		return sensor_conv_out
