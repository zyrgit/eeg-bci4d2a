#!/usr/bin/env python

import os, sys
# import subprocess
# import random, time
# import inspect
# import collections
# import math
# from shutil import copy2, move as movefile
# mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# sys.path.append(mypydir+"/mytools")

# from namehostip import get_my_ip
# from hostip import ip2tarekc
# from readconf import get_conf,get_conf_int,get_conf_float,get_list_startswith
# from logger import Logger

from scipy.fftpack import fft,rfft
import numpy as np

iprint =0

class FFT:
	def __init__(self,N=0,sampleRate=0):
		self.N = N  # N-pt fft
		self.spRate = sampleRate # sp/s

	def N_fft(self,x,N=0):
		if N>0:
			self.N=N
		x = np.asarray(x)#, dtype=np.float32) # if already, not copied.
		assert(self.N>=x.size)
		self.y = fft(x.reshape(1,x.size), n=self.N)[0]
		bad=-1
		if np.any(np.isnan(self.y.real)):
			bad=1
		if np.any(np.isnan(self.y.imag)):
			bad=1
		if bad==1:
			print("nan! when Nfft",self.N)
			print(x)
			sys.exit(0)
		return [self.y.real, self.y.imag]

	def sum_freq_band(self, bands=[]):
		if len(bands)<=0:
			bands=[[0.5,self.N/2.0]]
		res=[]
		for i in range(len(bands)):
			low_pos = int(float(bands[i][0])/self.spRate*self.N+0.5)
			high_pos= int(float(bands[i][1])/self.spRate*self.N+0.5)
			if iprint: print(low_pos,high_pos)
			re=0.0
			im=0.0
			for x in range(low_pos,high_pos+1):
				re += self.y.real[x]
				im += self.y.imag[x]
			res.append(re)
			res.append(im)
		return np.asarray(res)

	def run_fft(self, x):
		x = np.asarray(x)#, dtype=np.float32) # if already, not copied.
		self.y = fft(x.reshape(1,x.size))

	def get_re_im(self,x):
		self.run_fft( x)
		return [self.y.real, self.y.imag]

	def get_rfft_without_DC(self,x):
		x = np.asarray(x)#, dtype=np.float32) # if already, not copied.
		return rfft(x.reshape(1,x.size), n=self.N )[0,1:]

if __name__ == "__main__":
	f = FFT(N=128,sampleRate=128)
	print __file__
	# x=[1,2,3,4,5]
	# print(f.get_re_im(x)[0])
	# print(f.get_re_im(x)[1])
	x=range(128)
	print(f.N_fft(x)[0])
	print(f.N_fft(x)[1])
	# f.sum_freq_band([[1,4],[4,8],[8,14],[14,32],[32,64]])
	res=f.sum_freq_band([[0.5,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,31],[32,35],[36,39],[40,43],[44,49],[51,64]])
	print(res)
	# x=[[1],[2],[3],[4],[5]]
	print(f.get_rfft_without_DC(x)[99])

