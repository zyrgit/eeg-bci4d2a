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


def gru2(inputs, param_dict={}, train=True, name='gru2'): # [None, rnnDim]
	batch_size =param_dict["batch_size"]
	rnnDim =param_dict["rnnDim"]

	if train:
		reusing = False
	else:
		reusing = True

	with tf.variable_scope(name, reuse=reusing) as scope:

		inputs_shape = inputs.get_shape().as_list()
		# (None, time, dim)
		seq_length = tf.Variable(tf.ones([batch_size, inputs_shape[1] ]), name="seq_length", trainable=False)
		seq_length = tf.reduce_sum(seq_length, axis=1, keepdims=False)
		seq_length = tf.cast(seq_length, tf.int32)

		avgNum = tf.Variable(tf.ones([batch_size, inputs_shape[1], rnnDim]), dtype=tf.float32, name="avgNum", trainable=False)
		avgNum = tf.reduce_sum(avgNum, axis=1, keepdims=False)


		# gru_cell1 = tf.nn.rnn_cell.GRUCell(rnnDim)
		gru_cell1 = tf.contrib.rnn.GRUCell(rnnDim)
		if train:
			gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)
		gru_cell2 = tf.contrib.rnn.GRUCell(rnnDim)
		if train:
			gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

		cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2]) #([gru_cell1])  #
		init_state = cell.zero_state(batch_size, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, inputs, sequence_length=seq_length, initial_state=init_state, time_major=False)
		cell_output_shape = cell_output.get_shape().as_list()
		if iprint: print("cell_output_shape",cell_output_shape)
		# print("final_stateTuple len",len(final_stateTuple))

		sum_cell_out = tf.reduce_sum(cell_output, axis=1, keepdims=False) #*mask
		avg_cell_out = sum_cell_out/avgNum
		# [None, rnnDim]

	return avg_cell_out

