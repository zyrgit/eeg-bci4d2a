#!/usr/bin/env python
# cssh -l zhao97 tarekc26 tarekc04 tarekc15 tarekc28 tarekc14 tarekc16 tarekc10 tarekc08 tarekc40 tarekc18 tarekc17 tarekc34 tarekc07 tarekc19 tarekc12

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
print(tf.__version__)

import os, sys, getpass
import subprocess
import random, time
import inspect, glob
import copy
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
sys.path.append(mypydir+"/mytools")
import collections
import math
import random
import zipfile
import numpy as np
np.random.seed(1)

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from namehostip import get_my_ip
from hostip import ip2tarekc,tarekc2ip
from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
from logger import Logger
from util import read_lines_as_list,read_lines_as_dic
from inputClass import inputClass,FeedInput

layers = tf.contrib.layers 
activation_layer = tf.nn.elu

iprint = 1

HomeDir = os.path.expanduser("~")
User = getpass.getuser()

if HomeDir.endswith("shu"):
	WORKDIR = os.path.expanduser("~")
elif HomeDir.endswith("srallap"):
	WORKDIR = os.path.expanduser("~")+"/eeg"

configfile = "conf.txt"
confs = "conf/conf-%s.txt"

ind2trainSubj = {
0: [0], #[0,1,2], # 0:.42,1:.34,2:.28
1: [1], #[0,1,2,3,4], # 0:.53,1:.87,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.69,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,1,2,3,4,5,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.48,2:.66,3:.57,4:.82,5:.47,6:.80,7:.77,8:.68
}
ind2testSubj = {
0: [0], #[0,1,2], # 0:.42,1:.34,2:.28
1: [1], #[0,1,2,3,4], # 0:.53,1:.87,2:.48,3:.51,4:.49
2: [0], #[0], # 0:.69,
3: [1], #[0,1,2], # 0:.29,1:.42,2:.32 
4: [0,1,2,3,4,5,6,7,8], #[0,1,2,3,4,5,6,7,8], # 0:.59,1:.48,2:.66,3:.57,4:.82,5:.47,6:.80,7:.77,8:.68
}
ind2dataName = {
0:'bci3d3a', 
1:'bci3d4a', 
2:'bci3d4c', 
3:'bci3d5', 
4:'bci4d2a', 
}
dataName2ind={}
for k,v in ind2dataName.items():
	dataName2ind[v]=k

My_IP = get_my_ip()
lg = Logger(tag=__file__.split("/")[-1])
lg.lg_list([__file__])

from_conf = get_conf_int(configfile,"from_conf")
from_tasklist = get_conf_int(configfile,"from_tasklist")

use_cache = get_conf_int(configfile,"use_cache")
if use_cache:
	try:
		import memcache # https://www.sinacloud.com/doc/_static/memcache.html
	except:
		use_cache = 0
if use_cache:
	CACHE_SIZE = 100*1024*1024*1024 # 10G
	CACHE_TIME = 1000000#2weeks
	cache_place = get_conf(configfile,"cache_place")
	My_IP = get_my_ip()
	if iprint: print("cache loc "+cache_place)
	if cache_place=='local':
		cache_prefix = get_conf(configfile,"cache_prefix")
		mc = memcache.Client(['127.0.0.1:11211'], server_max_value_length=CACHE_SIZE)

	elif cache_place=='servers':
		datasets = get_list_startswith(configfile,"datasets")
		dataindex = 4 #int(sys.argv[1])
		dataname = datasets[dataindex]
		conf = confs%dataname
		cache_prefix = get_conf(conf,"cache_prefix")
		servers= get_conf(configfile,"cache_servers").split(",")
		if iprint: print("cache servers",servers)
		if My_IP.startswith("172.22"):
			ips = [tarekc2ip[host]+":11211" for host in servers]
		else:
			ips = [host+":11211" for host in servers]
		mc = memcache.Client(ips, server_max_value_length=CACHE_SIZE)

	elif cache_place=="lookup":
		mcdir = WORKDIR+"/mc/"
		datasets = get_list_startswith(configfile,"datasets")
		dataindex = 4 #int(sys.argv[1])
		dataname = datasets[dataindex]
		conf = confs%dataname
		cache_prefix = get_conf(conf,"cache_prefix")

		flist = glob.glob(mcdir+"/cache_server*")
		servers = []
		for i in range(len(flist)):
			serverip = read_lines_as_list(flist[i])
			serverip=serverip[0]
			servers.append(serverip)

			tmp=memcache.Client([serverip+":11211"])
			ret = tmp.set("tmp", 1 , time=10)
			val = tmp.get("tmp")
			if not (ret and val):
				print("-- Cache server fail: "+flist[i],serverip)
				mc_ind = flist[i].split(".")[0].lstrip(mcdir+"/cache_server")
				print("-- Run this cmd at login node:")
				print("jbsub -interactive -queue x86_7d -mem 40g sh "+WORKDIR+"/memcached/start.sh "+mc_ind)
				sys.exit(0)

		if iprint: print(__file__.split("/")[-1],"cache_prefix=",cache_prefix,dataname,servers)
		mc = memcache.Client([host+":11211" for host in servers], server_max_value_length=CACHE_SIZE)

	def mc_get(subj, key, pref=cache_prefix):
		return mc.get(pref+"s%d"%subj+key)
	def mc_set(subj, key, val, time=CACHE_TIME, pref=cache_prefix):
		return mc.set(pref+"s%d"%subj+key, val , time=time)

	ret = mc_set(0,"try_if_it_is_working", 1, time=10)
	val = mc_get(0,"try_if_it_is_working")
	if not (ret and val):
		print("memcache not working, exit")
		sys.exit(0)

MetaDir = get_conf(configfile,"metafolder")
DataDir = get_conf(configfile,"datafolder")
convertToSampleRate = get_conf_int(configfile,"convertToSampleRate")
mc_train_da_str = get_conf(configfile,"mc_train_da_str")
mc_train_lb_str = get_conf(configfile,"mc_train_lb_str")
mc_test_da_str = get_conf(configfile,"mc_test_da_str")
mc_test_lb_str = get_conf(configfile,"mc_test_lb_str")

batch_size = get_conf_int(conf,"batch_size")
regularize_coef=get_conf_float(conf,"regularize_coef")

scatter_mc = get_conf_int(conf,"scatter_mc")
numClasses = get_conf_int(conf,"numClasses")
stateStep = get_conf_int(conf,"stateStep")
nUseChannels = get_conf_int(conf,"nUseChannels")
secondPerFrame = get_conf_float(conf,"secondPerFrame")
height = nUseChannels
width = int(convertToSampleRate*secondPerFrame)

lg.lg_list(["regularize_coef=",regularize_coef])
lg.lg_list(["batch_size=",batch_size])
lg.lg_list(["stateStep=",stateStep])
lg.lg_list(["nUseChannels=",nUseChannels])
lg.lg_list(["secondPerFrame=",secondPerFrame])
lg.lg_list(["width=",width])
lg.lg_list(["height=",height])


SCOPE_CONV = "conv%d"
SCOPE_NORM = "norm%d"
SCOPE_DROP = "drop%d"
scope_cnt = 0
def scnt(inc=0, reset=False):
	global scope_cnt
	if reset:
		scope_cnt=0
	scope_cnt+=inc
	return scope_cnt


CONV_KEEP_PROB = 0.5

convNum1 = 60
convNum2 = 60
convNum3 = 60
convNum4 = 90
convNum5 = 120
convNum6 = 0
input_proj_dim = 30
fully_out_dim = 95
lg.lg_list(["convNum1=",convNum1])
lg.lg_list(["convNum2=",convNum2])
lg.lg_list(["convNum3=",convNum3])
lg.lg_list(["convNum4=",convNum4])
lg.lg_list(["convNum5=",convNum5])
lg.lg_list(["input_proj_dim=",input_proj_dim])
lg.lg_list(["CONV_KEEP_PROB=",CONV_KEEP_PROB])
lg.lg_list(["fully_out_dim=",fully_out_dim])


data1 = tf.placeholder(tf.float32, shape=[batch_size,stateStep,height,width])
labels = tf.placeholder(tf.float32, shape=[batch_size,numClasses])
global_step = tf.Variable(0, trainable=False)

is_training = True

from cnn2d import cnn2d,cnn2d_pool,cnn_pool_split
from cnn3d import cnn3d,cnn3d_dilated,cnn3d_pool_split
param_dict={}
param_dict["convNum1"]=convNum1
param_dict["convNum2"]=convNum2
param_dict["convNum3"]=convNum3
param_dict["convNum4"]=convNum4
param_dict["convNum5"]=convNum5
param_dict["convNum6"]=convNum6
param_dict["CONV_KEEP_PROB"]=CONV_KEEP_PROB
param_dict["OUT_DIM"]=fully_out_dim
cnn_param_dict = param_dict

# inputs = data1
inputs = tf.transpose(data1, perm=[0,1,3,2]) # [None,1,width,height]
with tf.variable_scope('input_layer',reuse=False):
	inputs = layers.fully_connected(inputs, input_proj_dim, activation_fn=activation_layer, scope='input')
	inputs_shape = inputs.get_shape().as_list()
	print('input transpose fully_connected',inputs_shape)
inputs = tf.transpose(inputs, perm=[0,1,3,2]) # [None,1,height,width]
inputs=tf.squeeze(inputs)
cnn_out = cnn2d_pool(inputs, cnn_param_dict, is_training) #(None, 100)

last_layer = cnn_out
last_layer_dim=last_layer.get_shape().as_list()[-1]

with tf.variable_scope('last_layer'):
	out_W = tf.get_variable("out_W", shape=[last_layer_dim, numClasses], initializer=tf.contrib.layers.variance_scaling_initializer())

logits = tf.matmul(last_layer, out_W)
predict = tf.argmax(logits, axis=1)
loss_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))


# ------------------- test data ------------------dropout false ---------
data1_t = tf.placeholder(tf.float32, shape=[batch_size,stateStep,height,width])

is_training = False

# inputs = data1_t
inputs = tf.transpose(data1_t, perm=[0,1,3,2]) # [None,1,width,height]
with tf.variable_scope('input_layer',reuse=True):
	inputs = layers.fully_connected(inputs, input_proj_dim, activation_fn=activation_layer, scope='input')
	inputs_shape = inputs.get_shape().as_list()
	print('input transpose fully_connected',inputs_shape)
inputs = tf.transpose(inputs, perm=[0,1,3,2]) # [None,1,height,width]
inputs = tf.squeeze(inputs) 

cnn_out = cnn2d_pool(inputs, cnn_param_dict, is_training) #(None, ens, 100)

last_layer = cnn_out
logits = tf.matmul(last_layer, out_W)
predict_t = tf.argmax(logits, axis=1)
loss_ent_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))


# ------------------- summary ----------------------
t_vars = tf.trainable_variables()
regularizers = 0.0
for var in t_vars:
	regularizers += tf.nn.l2_loss(var)
loss = loss_ent+ regularize_coef * regularizers


total_parameters = 0
for variable in tf.trainable_variables():
	# shape is an array of tf.Dimension
	shape = variable.get_shape().as_list()
	variable_parametes = 1
	for dim in shape:
		variable_parametes *= dim
	print(variable.name, shape, variable_parametes)
	total_parameters += variable_parametes
print("total_parameters",total_parameters)
lg.lg_list(["total_parameters=",total_parameters])


## -------------------------- data feed below -----------------------------
if use_cache:
	trainSubj = ind2trainSubj[ dataName2ind[dataname] ]
	testSubj = ind2testSubj[ dataName2ind[dataname] ]
	if iprint: print("trainSubj",trainSubj)
	if iprint: print("testSubj",testSubj)

	intrain_all_data1= []
	intrain_all_label= []
	intest_all_data1=[]
	intest_all_label=[]

	storelist = [intrain_all_data1,intrain_all_label,intest_all_data1,intest_all_label]
	storekey = [mc_train_da_str,mc_train_lb_str,mc_test_da_str,mc_test_lb_str]

	for s in trainSubj:
		mc_num_per_key=mc_get(s,"mc_num_per_key")
		if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)
		for i in [0,1]:
			da= storelist[i]
			sz= mc_get(s,storekey[i]+"-max")
			if iprint: print(s,storekey[i]+"-max",sz)
			assert(sz)
			cnt=0
			while cnt<=sz:
				ret = mc_get(s,storekey[i]+"%d"%cnt)
				cnt+=mc_num_per_key
				if ret is None:
					print(__file__.split("/")[-1], "memcache fail ... exit")
					sys.exit(0)
				else:
					if cnt%2000==0:
						print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
				for item in ret:
					da.append(item)
	for s in testSubj:
		mc_num_per_key=mc_get(s,"mc_num_per_key")
		if iprint: print(__file__.split("/")[-1],"mc_num_per_key:",mc_num_per_key)
		for i in [2,3]:
			da= storelist[i]
			sz= mc_get(s,storekey[i]+"-max")
			if iprint: print(s,storekey[i]+"-max",sz)
			assert(sz)
			cnt=0
			while cnt<=sz:
				ret = mc_get(s,storekey[i]+"%d"%cnt)
				cnt+=mc_num_per_key
				if ret is None:
					print(__file__.split("/")[-1], "memcache fail ... exit")
					sys.exit(0)
				else:
					if cnt%2000==0:
						print(__file__.split("/")[-1], "memcache read %s"%(storekey[i]+"%d"%cnt))
				for item in ret:
					da.append(item)

	# need sp size % batch == 0:
	size = len(intrain_all_label)
	size_round = size//batch_size * batch_size
	size_add = batch_size - size + size_round
	print("size",size,"size_add",size_add)
	
	scatter_ratio = height//scatter_mc
	print("scatter_mc",scatter_mc,scatter_ratio,height)

	for i in range(size_add):
		ind = np.random.randint(0,high=size)
		if scatter_mc>0:
			for j in range(ind*scatter_ratio, (ind+1)*scatter_ratio):
				intrain_all_data1.append(copy.deepcopy(intrain_all_data1[j]))
		else:
			intrain_all_data1.append(copy.deepcopy(intrain_all_data1[ind]))
		intrain_all_label.append(copy.deepcopy(intrain_all_label[ind]))

	size = len(intest_all_label)
	size_round = size//batch_size * batch_size
	if size>size_round:
		size_add = batch_size - size + size_round
	else:
		size_add=0
	print("size",size,"test size_add",size_add)

	for i in range(size_add):
		ind = np.random.randint(0,high=size)
		if scatter_mc>0:
			for j in range(ind*scatter_ratio, (ind+1)*scatter_ratio):
				intest_all_data1.append(copy.deepcopy(intest_all_data1[j]))
		else:
			intest_all_data1.append(copy.deepcopy(intest_all_data1[ind]))
		intest_all_label.append(copy.deepcopy(intest_all_label[ind]))
		
	intrain_all_data1=np.asarray(intrain_all_data1)
	intrain_all_label=np.asarray(intrain_all_label)
	intest_all_data1=np.asarray(intest_all_data1)
	intest_all_label=np.asarray(intest_all_label)

	if scatter_mc>0:
		tmp_shape = intrain_all_data1.shape # [None*?, step, scatter_mc, time]
		intrain_all_data1 = np.reshape(intrain_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], height, tmp_shape[3]))
		tmp_shape = intest_all_data1.shape # [None*?, step, scatter_mc, time]
		intest_all_data1 = np.reshape(intest_all_data1,(tmp_shape[0]//scatter_ratio, tmp_shape[1], height, tmp_shape[3]))

print(__file__.split("/")[-1], dataname,trainSubj)
print(__file__.split("/")[-1], dataname,testSubj)
print(__file__.split("/")[-1],"train data:")
print(__file__.split("/")[-1], intrain_all_data1.shape)
print(__file__.split("/")[-1], intrain_all_label.shape)
print(__file__.split("/")[-1], intrain_all_label[0:15])
print(__file__.split("/")[-1],"test data:")
print(__file__.split("/")[-1], intest_all_data1.shape)
print(__file__.split("/")[-1], intest_all_label.shape)
print(__file__.split("/")[-1], intest_all_label[0:15])
train_size = intrain_all_data1.shape[0]
test_size = intest_all_data1.shape[0]
assert(intrain_all_data1.shape[0]>0)
assert(intrain_all_data1.shape[0]==intrain_all_label.shape[0])
assert(intest_all_data1.shape[0]>0)
assert(intest_all_data1.shape[0]==intest_all_label.shape[0])

trainFeed = FeedInput([intrain_all_data1, intrain_all_label],batch_size)
trainFeed.shuffle_all()
testFeed = FeedInput([intest_all_data1, intest_all_label],batch_size)
testFeed.shuffle_all()
num_batches = trainFeed.get_num_batches()
print("num_batches per epoch",num_batches)

## -------------------------- data feed finish -----------------------------

init_lr = get_conf_float(conf,"init_lr")
end_lr = get_conf_float(conf,"end_lr")
nepoch = get_conf_int(conf,"nepoch") 
total_steps = nepoch*num_batches
decay_steps = num_batches/4.0
weight_decay_rate = (end_lr/init_lr)**(decay_steps/total_steps)
lg.lg_list(["init_lr=",init_lr])
lg.lg_list(["end_lr=",end_lr])
lg.lg_list(["nepoch=",nepoch])
lg.lg_list(["total_steps=",total_steps])
lg.lg_list(["decay_steps=",decay_steps])
lg.lg_list(["weight_decay_rate=",weight_decay_rate])
lg.flush()
print("nepoch=",nepoch,"total_steps",total_steps,"decay_steps",decay_steps,"weight_decay_rate",weight_decay_rate)

lr = tf.train.exponential_decay(init_lr, global_step, decay_steps, weight_decay_rate, staircase=True)

optimizer = tf.train.RMSPropOptimizer(lr)
gvs = optimizer.compute_gradients(loss, var_list=t_vars)
capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
discOptimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step) # inc global_step 
saver = tf.train.Saver() # must be in graph 

iteration = 0
testcnt = 2

with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)) as sess:
	print("graph_def",sess.graph_def.ByteSize()/1024.0,"KB")

	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	while trainFeed.get_epoch()<nepoch:
		iteration+=1

		batch_inputs, batch_labels = trainFeed.get_batch()
		feed_dict = {data1: batch_inputs, labels: batch_labels}

		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, labels, predict], feed_dict=feed_dict)

		if iteration % 50 == 0:
			lr_val = sess.run([lr])
			_label = np.argmax(_trainY, axis=1)
			_accuracy = np.mean(_label == _predict)
			print('train loss', lossV,'train accuracy', _accuracy)
			print("epoch",trainFeed.get_epoch(),"batch",iteration,"lr",lr_val)
			trainloss = lossV
			trainacc = _accuracy


		if trainFeed.get_epoch()>=testcnt:
			testcnt+=2
			dev_accuracy = []
			dev_cross_entropy = []
			for eval_idx in xrange(int(test_size/batch_size)):
				batch_inputs, batch_labels = testFeed.get_batch()
				feed_dict = {data1_t: batch_inputs, labels: batch_labels}

				eval_loss_v, _trainY, _predict = sess.run([loss_ent_t, labels, predict_t], feed_dict=feed_dict)
				_label = np.argmax(_trainY, axis=1)
				_accuracy = np.mean(_label == _predict)
				dev_accuracy.append(_accuracy)
				dev_cross_entropy.append(eval_loss_v)
			print(dataname,numClasses,'--- test l=',np.mean(dev_cross_entropy),"a=",np.mean(dev_accuracy))
			testloss = np.mean(dev_cross_entropy)
			testacc = np.mean(dev_accuracy)
			lg.lg_list(["epoch=",trainFeed.get_epoch(),"trainloss=",trainloss,"trainacc",trainacc,"testloss",testloss,"testacc",testacc])
			lg.flush()


	dev_accuracy = []
	dev_cross_entropy = []
	for eval_idx in xrange(int(test_size/batch_size)):
		batch_inputs, batch_labels = testFeed.get_batch()
		feed_dict = {data1_t: batch_inputs, labels: batch_labels}

		eval_loss_v, _trainY, _predict = sess.run([loss_ent_t, labels, predict_t], feed_dict=feed_dict)
		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		dev_accuracy.append(_accuracy)
		dev_cross_entropy.append(eval_loss_v)
	print(dataname,numClasses,'----test loss',np.mean(dev_cross_entropy),"accuracy",np.mean(dev_accuracy))
	testloss = np.mean(dev_cross_entropy)
	testacc = np.mean(dev_accuracy)
	lg.lg_list(["Final testloss",testloss,"testacc",testacc])
	lg.flush()


	save_path = saver.save(sess, "./model.ckpt")
	print(My_IP,"Saved: %s" % (save_path))

print(__file__.split("/")[-1], dataname,trainSubj,testSubj)
print("cat "+lg.fnames[0])
