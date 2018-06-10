#!/usr/bin/env python

import os, sys
# import subprocess
import random, time
import inspect
# import collections
# import math
# from shutil import copy2, move as movefile
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir+"/mytools")

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from procSegment import procSegment

iprint =1

class inputClass:
	def __init__(self,flist,params): 
		if "MetaDir" in params.keys(): MetaDir = params["MetaDir"]
		if "dataProcDir" in params.keys(): self.dataProcDir = params["dataProcDir"]
		if "secondPerFrame" in params.keys(): secondPerFrame = params["secondPerFrame"]
		if "sampleRate" in params.keys(): sampleRate = params["sampleRate"]
		if "label2softmax" in params.keys(): self.label2softmax = params["label2softmax"]
		if "numRawChannels" in params.keys(): numRawChannels = params["numRawChannels"]
		if "shiftRatio" in params.keys(): shiftRatio = params["shiftRatio"]
		if "stateStep" in params.keys(): self.stateStep = params["stateStep"]
		if "dataname" in params.keys(): self.dataname = params["dataname"]
		if "scatter_mc" in params.keys(): self.scatter_mc = params["scatter_mc"]
		if "nUseChannels" in params.keys(): self.nUseChannels = params["nUseChannels"]

		self.meta_flist = []
		for fname in flist:
			if "/" in fname:
				self.fname = fname.split("/")[-1]
			elif "." in fname.rstrip(".txt"):
				print(__file__.split("/")[-1]," . cannot be in non-suffix !")
				sys.exit(0)
			else:
				self.fname = fname
			self.meta_flist.append(self.dataProcDir+"/"+self.dataname+"/"+MetaDir+ self.fname.split(".")[0]+"-sameLabel.txt")

		self.numRawChannels = numRawChannels # 22 chn
		self.global_count = 0.0
		self.inputflist = []
		self.params = params
		lowcut=1.0
		highcut=40

		if iprint: print(__file__.split("/")[-1],"numRawChannels",numRawChannels,"shiftRatio",shiftRatio,"sampleRate",sampleRate,"lowcut",lowcut,"highcut",highcut,"secondPerFrame",secondPerFrame,"stateStep",self.stateStep,"nUseChannels",self.nUseChannels)
		
		self.proc = procSegment(self.nUseChannels,shiftRatio,sampleRate,lowcut,highcut,secondPerFrame,self.params)

	def read_flist(self): # read fname of file seg same label 
		for flistname in self.meta_flist:
			with open(flistname,"r") as fd:
				for l in fd:
					l=l.strip()
					if len(l)>0:
						self.inputflist.append( l ) # already abs path.
		return self.inputflist

	def read_augmented_data_in_steps(self,steps,):
		self.all_data1=[]
		self.all_label=[]
		if self.inputflist == []:
			self.read_flist()
		cnt=0
		for fn in self.inputflist:
			if iprint>=1: 
				cnt+=1
				if cnt%100==0: 
					print(__file__.split("/")[-1], fn)
					print("augmented %d times"%(len(atime)-1))
			atime,alabel = self.proc.augment_file(fn)
			for i in range(len(atime)):  # for each augmented version
				dtime = atime[i] # contains steps
				dlabel = alabel[i]
				nstep = len(dtime)
				assert(nstep>=steps)
				if nstep==steps:
					self.all_data1.append( dtime ) 
					self.all_label.append( self.label2softmax[int(dlabel[0])] )
				else:
					for k in range(0,nstep+1-steps,steps):
						self.all_data1.append( dtime[k:k+steps] ) 
						self.all_label.append( self.label2softmax[int(dlabel[0])] )
		self.all_data1 = np.asarray(self.all_data1)
		scatter_mc=self.scatter_mc
		if scatter_mc>0:
			tmp_shape = self.all_data1.shape # [None, step, chn, time]
			self.all_data1 = np.reshape(self.all_data1,(tmp_shape[0]*tmp_shape[2]/scatter_mc, tmp_shape[1], scatter_mc, tmp_shape[3]))
		self.all_label = np.asarray(self.all_label)
		self.all_data_size = self.all_data1.shape[0]
		print(__file__.split("/")[-1], "all_data all_label shape: ")
		print(__file__.split("/")[-1], self.all_data1.shape)
		print(__file__.split("/")[-1], self.all_label.shape)

	def read_augmented_data_in_steps_td_fd(self,steps,):
		self.all_data1=[]
		self.all_data2=[]
		self.all_label=[]
		if self.inputflist == []:
			self.read_flist()
		cnt=0
		for fn in self.inputflist:
			if iprint>=1: 
				cnt+=1
				if cnt%100==0: 
					print(__file__.split("/")[-1], fn)
					print("augmented %d times"%(len(atime)-1))
			atime,afreq,alabel = self.proc.augment_file(fn)
			for i in range(len(atime)):  # for each augmented version
				dtime = atime[i] # contains steps
				dfreq = afreq[i]
				dlabel = alabel[i]
				nstep = len(dtime)
				assert(nstep==len(dfreq))
				assert(nstep>=steps)
				if nstep==steps:
					self.all_data1.append( dtime ) 
					self.all_data2.append( dfreq ) 
					self.all_label.append( self.label2softmax[int(dlabel[0])] )
				else:
					for k in range(0,nstep+1-steps,steps):
						self.all_data1.append( dtime[k:k+steps] ) 
						self.all_data2.append( dfreq[k:k+steps] ) 
						self.all_label.append( self.label2softmax[int(dlabel[0])] )
		self.all_data1 = np.asarray(self.all_data1)
		self.all_data2 = np.asarray(self.all_data2)
		self.all_label = np.asarray(self.all_label)
		self.all_data_size = self.all_data1.shape[0]
		print(__file__.split("/")[-1], "all_data all_label shape: ")
		print(__file__.split("/")[-1], self.all_data1.shape)
		print(__file__.split("/")[-1], self.all_data2.shape)
		print(__file__.split("/")[-1], self.all_label.shape)

	def get_a_sample(self,steps, size = 10): # mannual edit return sp sz, in for.
		sp_data1=[]
		sp_label=[]
		if self.inputflist == []:
			self.read_flist()
		spn = size
		for fn in self.inputflist:
			if iprint>=1: 
					print(__file__.split("/")[-1],"Sample", fn)
			atime,alabel = self.proc.augment_file(fn)
			for i in range(len(atime)):  # for each augmented version
				dtime = atime[i] # contains steps
				dlabel = alabel[i]
				nstep = len(dtime)
				assert(nstep>=steps)
				if nstep==steps:
					for s in range(spn):
						sp_data1.append( dtime ) 
						sp_label.append( self.label2softmax[int(dlabel[0])] )
				else:
					for k in range(0,nstep+1-steps,steps):
						for s in range(spn):
							sp_data1.append( dtime[k:k+steps] ) 
							sp_label.append( self.label2softmax[int(dlabel[0])] )
			break

		sp_data1 = np.asarray(sp_data1)
		scatter_mc=self.scatter_mc
		if scatter_mc>0:
			tmp_shape = sp_data1.shape # [None, step, chn, time]
			sp_data1 = np.reshape(sp_data1,(tmp_shape[0]*tmp_shape[2]/scatter_mc, tmp_shape[1], scatter_mc, tmp_shape[3]))
		sp_label = np.asarray(sp_label)
		print(__file__.split("/")[-1], "sample shape: ")
		print(__file__.split("/")[-1], sp_data1.shape)
		print(__file__.split("/")[-1], sp_label.shape)
		return sp_data1,sp_label

	def get_a_sample_td_fd(self,steps, size = 10): # mannual edit return sp sz, in for.
		sp_data1=[]
		sp_data2=[]
		sp_label=[]
		if self.inputflist == []:
			self.read_flist()
		spn = size
		for fn in self.inputflist:
			if iprint>=1: 
					print(__file__.split("/")[-1],"Sample", fn)
			atime,afreq,alabel = self.proc.augment_file(fn)
			for i in range(len(atime)):  # for each augmented version
				dtime = atime[i] # contains steps
				dfreq = afreq[i]
				dlabel = alabel[i]
				nstep = len(dtime)
				assert(nstep==len(dfreq))
				assert(nstep>=steps)
				if nstep==steps:
					for s in range(spn):
						sp_data1.append( dtime ) 
						sp_data2.append( dfreq ) 
						sp_label.append( self.label2softmax[int(dlabel[0])] )
				else:
					for k in range(0,nstep+1-steps,steps):
						for s in range(spn):
							sp_data1.append( dtime[k:k+steps] ) 
							sp_data2.append( dfreq[k:k+steps] ) 
							sp_label.append( self.label2softmax[int(dlabel[0])] )
			break

		sp_data1 = np.asarray(sp_data1)
		sp_data2 = np.asarray(sp_data2)
		sp_label = np.asarray(sp_label)
		print(__file__.split("/")[-1], "sample shape: ")
		print(__file__.split("/")[-1], sp_data1.shape)
		print(__file__.split("/")[-1], sp_data2.shape)
		print(__file__.split("/")[-1], sp_label.shape)
		return sp_data1,sp_data2,sp_label

	def shuffle_all(self,):
		self.all_data1, self.all_label = shuffle(self.all_data1,self.all_label,random_state=0)

	def shuffle_all2(self,):
		self.all_data1,self.all_data2, self.all_label = shuffle(self.all_data1,self.all_data2,self.all_label,random_state=0)

	def get_num_epoch(self,):
		return self.global_count/self.all_data_size




# -------------- memcached 
class FeedInput:
	def __init__(self,data_list,batch_size): 
		assert(len(data_list)<10) # data_list is lists of input modality, not too big. 
		self.datas = data_list # [data1, data2..., label,]
		assert(len(self.datas[0])==len(self.datas[1]))
		self.batch_size = batch_size
		self.index=0
		self.size = len(self.datas[0])
		self.count = 0

	def shuffle_all(self,):
		all_data1, all_label = shuffle(self.datas[0], self.datas[1], random_state=0)
		self.datas[0] = all_data1
		self.datas[1] = all_label

	def get_batch(self,dtype=np.float32): # return [ d1[0:batch], d2[0:batch], ... lb[0:batch]]
		res = []
		start = self.index
		end = start + self.batch_size
		over = max(0,end - self.size)
		end = min(end, self.size)
		for i in range(len(self.datas)):
			res.append(self.datas[i][start:end])
		if over>0:
			for i in range(len(self.datas)):
				res[i]= np.append(res[i], self.datas[i][0:over], axis=0)
		for i in range(len(self.datas)):
			res[i]=np.asarray(res[i],dtype=dtype)
		if over>0:
			self.index = over
		else:
			self.index = end
		self.count+=self.batch_size
		return res

	def get_epoch(self,):
		return float(self.count)/self.size

	def get_data_size(self,):
		return self.size
	def get_num_batches(self,):
		return float(self.size)/self.batch_size

if __name__ == "__main__":
	x1=np.asarray([1,2,3,4,5,6,7,8,9])
	x2=np.asarray([11,12,13,14,15,16,17,18,19])
	lb=np.asarray([1,1,0,0,1,1,0,0,1])
	a=FeedInput([x1,x2,lb],6)
	print(a.get_batch(),a.get_epoch())
	print(a.get_batch(),a.get_epoch())
	print(a.get_batch(),a.get_epoch())
	print(a.get_batch(),a.get_epoch())
	print(a.get_batch(),a.get_epoch())
	sys.exit(0)

	a = inputClass(["data.txt"],2,0,22)
	a.read_flist("./")
	a.read_all_data()
	a.shuffle_all2()
	# print(a.generate_batch2()[1])
	print(a.generate_batch2()[2])