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

iprint = 2

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

def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True, 
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)


def cnn_pool_split(inputs, param_dict={}, train=True, name='cnn_pool_split'): # out (None, ensenble, 100)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]
	OUT_DIM = param_dict["OUT_DIM"]
	if train:
		reusing = False
	else:
		reusing = True
	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None, 3, 25, 125) 
			# inputs = tf.transpose(inputs, perm=[0,2,1,3])
			# (None, 25,3, 125) 
			out_shape = inputs.get_shape().as_list()
			# inputs = tf.reshape(inputs,[out_shape[0],out_shape[1],out_shape[2]*out_shape[3]])
			# (None, 25,375) 
		elif len(inputs.get_shape().as_list())==3:
			# (None, 25,375)
			inputs = tf.expand_dims(inputs, axis=1)
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		# (None, 1,25,375) 
		inputs = tf.expand_dims(inputs, axis=4)
		# (None, 1,25,375,1) 
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())
		padding_cv = "VALID"
		padding_pool = "SAME"
		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]
		pool_stride = [1,1,1,1,1]

		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 1, 15],
						stride=[1, 1, 3], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 1, 25, 181, 64)
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
		# (None, 2, 25, 90, 64)

		padding_cv = "SAME"
		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 1, 4],
						stride=[1, 1, 2], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 2, 25, 44, 64)
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
		# (None, 4, 25, 22, 64)


		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, conv_shape[2], 1],
						stride=[1, 1, 1], padding="VALID", data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 4, 1, 22, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train, 
			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		padding_cv = "SAME"
		pool_size = 2
		pool_ksize = [1,1,1,pool_size,1]
		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 9, 1, 20, 64)
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
		# (None, 8, 1, 10, 64)


		#------ add dynamically -------
		while conv_shape[3]>20:
			conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 1, 3],
							stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
			# (None, ?, 1, ?, 64)
			conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
			conv = activation_layer(conv)
			conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) #(60)
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

		#------ above dynamically -------

		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 1, 3],
						stride=[1, 1, 1], padding=padding_cv, data_format='NDHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		# (None, 18, 1, 8, 64)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool3d(conv,pool_ksize,pool_stride,padding=padding_pool,data_format='NDHWC',name=SCOPE_POOL%scnt()) #(60)
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
		# (None, 16, 1, 4, 64)

		conv = tf.reshape(conv, [conv_shape[0], conv_shape[1], conv_shape[2]*conv_shape[3]*conv_shape[4]])
		conv_shape = conv.get_shape().as_list()
		print(scnt(1),conv_shape)
		#(None, 16, 256)

		conv = layers.fully_connected(conv, OUT_DIM, activation_fn=activation_layer, scope='output')
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0],1, conv_shape[2]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)

		return conv

def cnn2d_nopool_rawout(inputs, param_dict={}, train=True, name='cnn2d_nopool_rawout'):#->(None, chn, time, filt)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]
	if "OUT_DIM" in param_dict.keys(): OUT_DIM = param_dict["OUT_DIM"]
	if "last_dim" in param_dict.keys(): OUT_DIM = param_dict["last_dim"]
	if train:
		reusing = False
	else:
		reusing = True
	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None, 3, 25, 125) 
			inputs = tf.transpose(inputs, perm=[0,2,1,3])
			# (None, 25,3, 125) 
			out_shape = inputs.get_shape().as_list()
			inputs = tf.reshape(inputs,[out_shape[0],out_shape[1],out_shape[2]*out_shape[3]])
			# (None, 25,375) 
		elif len(inputs.get_shape().as_list())==3:
			# (None, 25,375)
			pass
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		inputs = tf.expand_dims(inputs, axis=3)
		# (None, 25,375,1) 
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())

		# pooling = [1,1,2,1]

		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 6],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 366, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) # (None, 25, 183, 16)

		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 3],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 178, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) # (None, 25, 89, 16)

		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, 3],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 86, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) # (None, 25, 53, 16)



		conv = layers.convolution2d(conv, convNum4, kernel_size=[conv_shape[1], 1],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 53, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)



		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, 3],stride=[1, 1], padding='SAME', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 53, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 3],stride=[1, 1], padding='SAME', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 27, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		# conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		# conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				# noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		
		return conv #(None, chn, t, filt)


def cnn2d_pool(inputs, param_dict={}, train=True, name='cnn2d_pool'): # out (None, 100)
	convNum1 =param_dict["convNum1"]
	convNum2 =param_dict["convNum2"]
	convNum3 =param_dict["convNum3"]
	convNum4 =param_dict["convNum4"]
	convNum5 =param_dict["convNum5"]
	CONV_KEEP_PROB = param_dict["CONV_KEEP_PROB"]
	if "OUT_DIM" in param_dict.keys(): OUT_DIM = param_dict["OUT_DIM"]
	if "last_dim" in param_dict.keys(): OUT_DIM = param_dict["last_dim"]

	if train:
		reusing = False
	else:
		reusing = True

	with tf.variable_scope(name, reuse=reusing) as scope:
		scnt(reset=True)
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None, 3, 25, 125) 
			inputs = tf.transpose(inputs, perm=[0,2,1,3])
			# (None, 25,3, 125) 
			out_shape = inputs.get_shape().as_list()
			inputs = tf.reshape(inputs,[out_shape[0],out_shape[1],out_shape[2]*out_shape[3]])
			# (None, 25,375) 
		elif len(inputs.get_shape().as_list())==3:
			# (None, 25,375)
			pass
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		inputs = tf.expand_dims(inputs, axis=3)
		# (None, 25,750,1) 
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())

		pooling = [1,1,2,1]

		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 15],stride=[1, 3], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 250, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) # (None, 25, 125, 16)


		conv = layers.convolution2d(conv, convNum1, kernel_size=[1, 4],stride=[1, 2], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 62, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) # (None, 25, 31, 16)



		conv = layers.convolution2d(conv, convNum3, kernel_size=[conv_shape[1], 1],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 31, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum4, kernel_size=[1, 3],stride=[1, 1], padding='SAME', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 30, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) 
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) #(None, 1, 15, 16)


		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 3],stride=[1, 1], padding='SAME', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 15, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv = tf.nn.max_pool(conv,pooling,pooling,padding="SAME",data_format='NHWC',name=SCOPE_POOL%scnt()) #(7)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape) #(None, 1, 7, 16)


		conv = tf.reshape(conv, [conv_shape[0], conv_shape[1]*conv_shape[2]*conv_shape[3]])
		conv_shape = conv.get_shape().as_list()
		print(scnt(1),conv_shape)
		#(None, 500)

		conv = layers.fully_connected(conv, OUT_DIM, activation_fn=activation_layer, scope='output')
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], conv_shape[1]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)

		return conv


def cnn2d(inputs, param_dict={}, train=True, name='cnn2d'): # out (None, 368, 16)
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
		if len(inputs.get_shape().as_list())==4:
			# inputs shape (None, 3, 25, 125) 
			inputs = tf.transpose(inputs, perm=[0,2,1,3])
			# (None, 25,3, 125) 
			out_shape = inputs.get_shape().as_list()
			inputs = tf.reshape(inputs,[out_shape[0],out_shape[1],out_shape[2]*out_shape[3]])
			# (None, 25,375) 
		elif len(inputs.get_shape().as_list())==3:
			# (None, 25,375)
			pass
		else:
			print(__file__.split("/")[-1],"wrong input dim!")
			sys.exit(0)
			
		inputs = tf.expand_dims(inputs, axis=3)
		# (None, 25,375,1) 
		print(__file__.split("/")[-1],"in shape",inputs.get_shape().as_list())


		conv = layers.convolution2d(inputs, convNum1, kernel_size=[1, 6],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 372, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum2, kernel_size=[1, 4],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 370, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum3, kernel_size=[1, 3],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 25, 368, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum4, kernel_size=[conv_shape[1], 1],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 368, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		print(scnt(),conv_shape)


		conv = layers.convolution2d(conv, convNum5, kernel_size=[1, 3],stride=[1, 1], padding='VALID', data_format='NHWC', scope=SCOPE_CONV%scnt(1),activation_fn=None)
		#(None, 1, 368, 16)
		conv = batch_norm_layer(conv, train, scope=SCOPE_NORM%scnt())
		conv = activation_layer(conv)
		conv_shape = conv.get_shape().as_list()
		conv = layers.dropout(conv, CONV_KEEP_PROB, is_training=train,
				noise_shape=[conv_shape[0], 1, 1, conv_shape[3]], scope=SCOPE_DROP%scnt())
		print(scnt(),conv_shape)
		

		conv = tf.squeeze(conv)
		#(None, 368, 16)

		return conv

