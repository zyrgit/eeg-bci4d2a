#!/usr/bin/env python

import os, sys
# import subprocess
# import random, time
import inspect
# from shutil import copy2, move as movefile
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir+"/mytools")

# from namehostip import get_my_ip
# from hostip import ip2tarekc
# from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith
from util import read_lines_as_dic
		
import numpy as np
import pandas as pd

iprint =1


class multiColSeqSeparate:
	def __init__(self,fname, params={}): # labelColNum not in use.
		if "MetaDir" in params.keys(): self.MetaDir = params["MetaDir"]
		if "DataDir" in params.keys(): self.DataDir = params["DataDir"]
		if "rawfolder" in params.keys(): self.rawfolder = params["rawfolder"]
		if "downsample" in params.keys(): self.downsample = params["downsample"]
		if "sep" in params.keys(): self.sep = params["sep"]
		if "numRawChannels" in params.keys(): numRawChannels = params["numRawChannels"]
		if "dataSetsDir" in params.keys(): self.dataSetsDir = params["dataSetsDir"]
		if "dataProcDir" in params.keys(): self.dataProcDir = params["dataProcDir"]
		if "nUseChannels" in params.keys(): nUseChannels = params["nUseChannels"]
		if "inputchannels" in params.keys(): inputchannels = params["inputchannels"]

		if "/" in fname:
			self.fname = fname.split("/")[-1]
		else:
			self.fname = fname

		if not os.path.exists(self.dataProcDir+"/"+self.MetaDir):
			os.makedirs(self.dataProcDir+"/"+self.MetaDir)
			if iprint>=1: print("mkdir",self.dataProcDir+"/"+self.MetaDir)
		else:
			if iprint>1: print("exists",self.dataProcDir+"/"+self.MetaDir)

		if not os.path.exists(self.dataProcDir+"/"+self.DataDir):
			os.makedirs(self.dataProcDir+"/"+self.DataDir)
			if iprint>=1: print("mkdir",self.dataProcDir+"/"+self.DataDir)
		else:
			if iprint>1: print("exists",self.dataProcDir+"/"+self.DataDir)

		self.lbcol = numRawChannels # label col, raw file col num: numRawChannels.

		
		if "0" not in inputchannels.keys():
			print("Warn, please check, first channel is 0 index!")
			sys.exit(0)
		self.useChannel2name={}
		for k,v in inputchannels.items():
			self.useChannel2name[int(k)]=v
		if iprint: print("useChannel2name",self.useChannel2name)
		assert(nUseChannels==len(self.useChannel2name))

		if iprint: print("downsample",self.downsample)

		if self.sep == "comma":
			self.sep=","
		elif self.sep == "space":
			self.sep=" "
		else:
			print("wrong sep !",self.fname)
			sys.exit(0)


	def load_data(self,):
		#not use 'sep', irregular spaces. none header, first line is data, wrong columns 
		if self.sep==" ":
			self.data = pd.read_csv(self.dataSetsDir+"/"+self.rawfolder+"/"+self.fname,header=None,delim_whitespace=True)
		else:
			self.data = pd.read_csv(self.dataSetsDir+"/"+self.rawfolder+"/"+self.fname,header=None,sep=self.sep)

		self.nRow=self.data.shape[0]
		self.nCol=self.data.shape[1]
		assert(self.nCol>=len(self.useChannel2name)+1) # plus label 


	def splitSeqUseLabel(self,):
		#df.iloc[1:2].to_csv(filename, index=False, header=False)
		fd = open(self.dataProcDir+"/"+self.MetaDir+"/"+self.fname.split(".")[0]+"-sameLabel.txt","w")
		tmp = sorted([[k,v] for k,v in self.useChannel2name.items()])
		header = [k[1] for k in tmp]
		colnlist = [k[0] for k in tmp]
		header.append("label")
		colnlist.append(self.lbcol)
		if iprint: print(header,colnlist)
		ind = 0
		startind = ind
		lastlb = self.data.iloc[startind,self.lbcol]
		try:
			while ind<self.nRow:
				if self.data.iloc[ind,self.lbcol]==lastlb:
					ind+=1
				else:
					for spindex in range(self.downsample):
						fname =self.dataProcDir+"/"+self.DataDir+ self.fname.split(".")[0]+"-%d"%(ind-1)+"-%d"%(spindex)+"-%d"%(lastlb)+".txt"
						self.data.iloc[startind+spindex:ind:self.downsample,colnlist].to_csv(fname,index=False,header=header)
						fd.write(fname+"\n")
					startind=ind
					lastlb = self.data.iloc[startind,self.lbcol]
			for spindex in range(self.downsample):
				fname =self.dataProcDir+"/"+self.DataDir+ self.fname.split(".")[0]+"-%d"%(ind-1)+"-%d"%(spindex)+"-%d"%(lastlb)+".txt"
				self.data.iloc[startind+spindex:ind:self.downsample,colnlist].to_csv(fname,index=False,header=header)
				fd.write(fname+"\n")
		except:
			print(fname)
			print(self.fname)
			sys.exit(0)
		fd.close()

