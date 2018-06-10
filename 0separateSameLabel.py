#!/usr/bin/env python

import os, sys, getpass
# import subprocess
# import random, time
import inspect, glob
# import collections
# import math
from shutil import copy2, move as movefile
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
sys.path.append(mypydir+"/mytools")

from namehostip import get_my_ip
from hostip import ip2tarekc
from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith,get_dic_startswith
from logger import Logger
from util import read_lines_as_list,read_lines_as_dic

from multiColSeqSeparate import multiColSeqSeparate

iprint = 1

HomeDir = os.path.expanduser("~")
User = getpass.getuser()

if HomeDir.endswith("shu"):
	WORKDIR = os.path.expanduser("~")
elif HomeDir.endswith("srallap"):
	WORKDIR = os.path.expanduser("~")+"/eeg"

configfile = "conf.txt"
from_conf = get_conf_int(configfile,"from_conf")
from_tasklist = get_conf_int(configfile,"from_tasklist")
dataSetsDir = get_conf(configfile,"dataSetsDir"+User)
dataProcDir = get_conf(configfile,"dataProcDir"+User)
rawfolder = get_conf(configfile,"rawfolder")
metafolder= get_conf(configfile,"metafolder")
datafolder= get_conf(configfile,"datafolder")

# datasets = get_list_startswith(configfile,"datasets")
datasets = ["bci4d2a"]
datapaths = [dataSetsDir+"/"+tmp+"/" for tmp in datasets]
if iprint: print(datapaths)

for i in range(len(datapaths)):
	confsuffix = datasets[i]
	path = datapaths[i]
	conf = conf = "conf/conf-"+confsuffix+".txt"

	iflist = glob.glob(path+"/"+rawfolder+"/*")

	params={}
	params["MetaDir"] = metafolder
	params["DataDir"] = datafolder
	params["rawfolder"] = rawfolder
	params["downsample"] = get_conf_int(conf,"downsample")
	params["sep"] = get_conf(conf,"sep")
	params["numRawChannels"] = get_conf_int(conf,"numRawChannels")
	params["dataSetsDir"] = dataSetsDir+"/"+confsuffix+"/"
	params["dataProcDir"] = dataProcDir+"/"+confsuffix+"/"
	params["nUseChannels"] = get_conf_int(conf,"nUseChannels")
	params["inputchannels"] = read_lines_as_dic("conf/inputchannels-%s.txt"%confsuffix,",")
	print(confsuffix,params)

	# sys.exit(0)
	for fn in iflist:
		if iprint: print("cutting",fn)

		a = multiColSeqSeparate( fn,params=params )
		a.load_data()
		a.splitSeqUseLabel()

	



