#!/usr/bin/env python

import os, sys
import subprocess
import random, time
import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
from namehostip import get_my_ip
from hostip import ip2tarekc
import datetime
import glob
from shutil import copy2, move as movefile

iprint = 1
iprintverb =0
folder  = "log/"
fnamePrefix = "%slog-"%folder

class Logger: 
	def __init__(self ,tag=""):
		self.lg_index=0
		self.my_ip = get_my_ip()
		if not os.path.exists(folder):
			os.makedirs(folder)
		try:
			self.my_tname = tag+ ip2tarekc[self.my_ip]
		except:
			self.my_tname = tag+ self.my_ip.split(".",2)[-1]
			if iprint>=2: print(self.my_tname)
		self.fd_list=[]
		self.fnames =[]
		self.freshness = 0 # latest log always named 0: log-tarekc-`0`-date-txt
		tmp = glob.glob(folder+"log*") # log/log*
		for fn in tmp:
			try:
				st = fn.replace(fnamePrefix+self.my_tname+"-","").split("-",1)
				ind = int(st[0])+1
				newfn = fnamePrefix+self.my_tname+"-%d-"%ind+st[-1]
				movefile(fn,newfn)
				if iprint>=2: print(fn,newfn)
			except:
				pass
		fmain = fnamePrefix+self.my_tname+"-0-"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".txt"
		if iprint>=1: print("create",fmain)
		fd = open(fmain,"w")
		self.fd_list.append(fd)
		self.fnames.append(fmain)
		self.lg(self.my_tname)
		self.lg(self.my_ip)
		self.lg(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		self.lg(time.time())
		self.lg("\n")

	def lg(self, st, i=-1):
		if i<0:
			i=self.lg_index
		st=str(st)
		if not st.endswith("\n"):
			st=st+"\n"
		self.fd_list[i].write(st)

	def overwrite(self,st,i=-1):
		if i<0:
			i=self.lg_index
		self.fd_list[i].close()
		self.fd_list[i] = open(self.fnames[i],"w")
		self.lg(self.my_tname,i)
		self.lg(self.my_ip,i)
		self.lg(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i)
		self.lg(time.time(),i)
		self.lg("\n",i)
		self.lg(st,i)

	def lg_new(self, st=""):
		ind = len(self.fd_list) # add suffix ind instead of inc -0-
		fn =fnamePrefix+self.my_tname+"-0-"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"-"+str(ind)+".txt"
		self.fd_list.append(open(fn,"w"))
		self.fnames.append(fn)
		if iprint>=1: print("create2",fn)
		self.lg(self.my_tname,ind)
		self.lg(self.my_ip,ind)
		self.lg(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),ind)
		self.lg(time.time(),ind)
		self.lg("\n",ind)
		if st!="":
			self.lg(st,ind)
		return ind

	def set_lg_index(self,ind):
		if ind>=0:
			self.lg_index=ind

	def lg_list(self,ls,i=-1):
		st=""
		for x in ls:
			st = st+ str(x) + " "
		self.lg(st,i)
	def lg_dict(self,dic,i=-1):
		for k,v in dic.items():
			self.lg(str(k)+" = "+str(v),i)
	
	def flush(self,):
		for fd in self.fd_list:
			fd.flush()
	def print_file_names(self,):
		for fn in self.fnames:
			print(fn)
	def __del__(self):
		for fd in self.fd_list:
			fd.close()

if __name__ == "__main__":
	l = Logger()
	l.lg("hello")
	l.lg(time.time())
	ind  = l.lg_new("another msg")
	l.lg("in another",ind)
	l.set_lg_index(ind)
	l.overwrite("overwrite another")
	l.set_lg_index(l.lg_new("3rd"))
	l.lg("")
	l.lg_list(["ads",23,3,"--"])
	l.set_lg_index(ind)
	l.lg_dict({1:2,"d":3})
