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


def clockworkrnn(inputs, param_dict={}, train=True): #,label_dim

	num_hidden =param_dict["num_hidden"]
	num_output =param_dict["num_output"]
	batch_size =param_dict["batch_size"]
	clockwork_periods =param_dict["clockwork_periods"][:]
	period_to_size = copy.deepcopy(param_dict["period_to_size"])

	total_sz = 0
	for k,v in period_to_size.items():
		total_sz+=v
	size_per_bin = float(num_hidden/total_sz)
	total_sz = 0
	for k in clockwork_periods:
		total_sz+=period_to_size[k]
		period_to_size[k] = int(size_per_bin * total_sz)

	if iprint: print(__file__.split("/")[-1],"#hidden",num_hidden)
	if iprint: print(__file__.split("/")[-1],"period_to_size",period_to_size)
	if train:
		reusing = False
		ind = 0
	else:
		reusing = True
		ind = 1
	
	with tf.variable_scope('clockworkrnn',reuse = reusing) as scope: # , reuse=reuse
		# shape_type=1 # (None, 25ch, 368t, 4filt)
		shape_type=2 # (None, 368t, 16filt)

		if shape_type == 1 : 
			time_inputs = tf.transpose(inputs, perm=[0,2,1,3])
			out_shape = time_inputs.get_shape().as_list()
			# (None, time, chn, filt) 
			time_inputs =tf.reshape(time_inputs,[out_shape[0],out_shape[1],out_shape[2]*out_shape[3]])
			# (None, time, chn*filt)
		elif shape_type ==2:
			time_inputs = inputs
			out_shape = time_inputs.get_shape().as_list()
			# (None, time, filt) 
			assert(len(out_shape)==3)
		else:
			print(__file__.split("/")[-1],"wrong in shape !")
			sys.exit(0)
		
		out_shape = time_inputs.get_shape().as_list()
		print(__file__.split("/")[-1],"time_inputs shape",out_shape)
		# (None, 375, 25*4)
		inDim = out_shape[-1]
		# label_dim = labels.get_shape().as_list()[-1]

		# Weight and bias initializers
		initializer_weights = tf.contrib.layers.variance_scaling_initializer()
		initializer_bias    = tf.constant_initializer(0.0)
		# Activation functions of the hidden and output state
		activation_hidden = tf.tanh
		activation_output = tf.nn.elu

		# Split into list of tensors, one for each timestep
		num_steps=out_shape[1]
		x_list = [tf.squeeze(x,axis=[1]) for x in tf.split(axis=1,num_or_size_splits=num_steps,value=time_inputs,name="inputs_list")]
		
		# Mask for matrix W_I to make sure it's upper triangular
		clockwork_mask =tf.constant(np.triu(np.ones([num_hidden, num_hidden])), dtype=tf.float32, name="mask")
		with tf.variable_scope("input"):
			input_W = tf.get_variable("W", shape=[inDim, num_hidden], initializer=initializer_weights)    # W_I
			input_b = tf.get_variable("b", shape=[num_hidden], initializer=initializer_bias)              # b_I
		with tf.variable_scope("hidden"):
			hidden_W = tf.get_variable("W", shape=[num_hidden, num_hidden], initializer=initializer_weights)  # W_H
			hidden_W = tf.multiply(hidden_W, clockwork_mask)  # => upper triangular matrix                    # W_H
			hidden_b = tf.get_variable("b", shape=[num_hidden], initializer=initializer_bias)                 # b_H
		with tf.variable_scope("output"):
			output_W = tf.get_variable("W", shape=[num_hidden, num_output], initializer=initializer_weights)  # W_O
			output_b = tf.get_variable("b", shape=[num_output], initializer=initializer_bias)                 # b_O
			# out_W = tf.get_variable("out_W", shape=[num_output, label_dim], initializer=initializer_weights)

		with tf.variable_scope("clockwork_cell") as scope2:
			# Initialize the hidden state of the cell to zero (this is y_{t_1})
			# state = tf.get_variable("state", shape=[batch_size, num_hidden], initializer=tf.zeros_initializer(), trainable=False)
			state = tf.Variable(tf.zeros([batch_size, num_hidden]), name="state%d"%ind, trainable=False)

			for time_step in range(num_steps):
				# Only initialize variables in the first step
				# if time_step > 0: scope2.reuse_variables()
				# Find the groups of the hidden layer that are active
				group_index = 0
				for i in range(len(clockwork_periods)):
					# Check if (t MOD T_i == 0)
					if time_step % clockwork_periods[i] == 0:
						# group_index = i+1  # note the +1
						group_index = period_to_size[ clockwork_periods[i] ]
				# Compute (W_I*x_t + b_I)
				WI_x = tf.matmul(x_list[time_step], tf.slice(input_W, [0, 0], [-1, group_index]))
				WI_x = tf.nn.bias_add(WI_x, tf.slice(input_b, [0], [group_index]), name="WI_x")
				# Compute (W_H*y_{t-1} + b_H), note the multiplication of the clockwork mask (upper triangular matrix)
				hidden_W = tf.multiply(hidden_W, clockwork_mask)
				WH_y = tf.matmul(state, tf.slice(hidden_W, [0, 0], [-1, group_index]))
				WH_y = tf.nn.bias_add(WH_y, tf.slice(hidden_b, [0], [group_index]), name="WH_y")
				# Compute y_t = (...) and update the cell state
				y_update = tf.add(WH_y, WI_x, name="state_update")
				y_update = activation_hidden(y_update)
				# Copy the updates to the cell state
				state = tf.concat(axis=1, values=[y_update, tf.slice(state, [0, group_index], [-1,-1])])

			# Save the final hidden state
			final_state = state
			# Compute the output, y = f(W_O*y_t + b_O)
			fc = tf.matmul(final_state, output_W)
			fc = tf.nn.bias_add(fc, output_b)
			out_layer = activation_output(fc, name="activation_output")
			# predictions = tf.matmul(predictions, out_W)


	return out_layer

