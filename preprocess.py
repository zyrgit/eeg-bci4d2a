#!/usr/bin/env python

import os, sys
# import subprocess
# import random, time
import inspect
# import collections
# import math
# from shutil import copy2, move as movefile
# mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# sys.path.append(mypydir+"/mytools")

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter#, iirnotch
from scipy import interpolate

iprint =1

def gen_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def gen_highpass( cutfreq, fs, order=5):
	nyq = 0.5 * fs
	high = cutfreq / nyq
	b, a = butter(order, high, btype='highpass')
	return b, a

# def gen_notch( cutfreq, fs, q=35):
# 	nyq = 0.5 * fs
# 	cutfreq = cutfreq / nyq
# 	b, a = iirnotch(cutfreq, q)
# 	return b, a

class preProcessor:
	def __init__(self, sampleRate, lowcut, highcut, weight_decay=0.999, init_window_size=100): 
		self.lowcut =float(lowcut)
		self.highcut=float(highcut)
		self.sampleRate = sampleRate # Hz
		self.decay = weight_decay
		self.window = init_window_size
		if iprint>=2: print(self.sampleRate,self.lowcut,self.highcut,self.decay,self.window)
		self.b_band , self.a_band = gen_bandpass(self.lowcut, self.highcut, self.sampleRate)
		self.b_high , self.a_high = gen_highpass(self.lowcut, self.sampleRate)
		# self.b_not , self.a_not = gen_notch(50 , self.sampleRate)
		self.a0=None
		self.a1=None

	def change_sample_rate(self, x0, fromrate, torate):
		if self.a0 is None or len(self.a0)!=len(x0):
			self.a0 = np.arange(0,len(x0))
			if iprint>=2: print("x0 len", len(self.a0))

		newlen = int(float(len(x0)-1)/fromrate*torate) # e.g. 2pts, 1 unit of time, need -1 
		if self.a1 is None or len(self.a1)!=newlen:
			slot = 1.0/torate*fromrate
			newmax = (newlen+0.01)*slot
			self.a1 = np.arange(0, newmax , slot)
			if iprint>=2: print("x1 len", len(self.a1))

		if torate>=fromrate:
			f = interpolate.interp1d(self.a0,x0,kind="linear") # "quadratic" cubic
		else:
			f = interpolate.interp1d(self.a0,x0,kind="quadratic") # "quadratic" cubic
		return f(self.a1)

	def filter_not(self,x):
		x = np.asarray(x).reshape(1,-1)
		return lfilter(self.b_not,self.a_not, x, axis=-1)
	def filter_high(self,x):
		x = np.asarray(x).reshape(1,-1)
		return lfilter(self.b_high,self.a_high, x, axis=-1)
	def filter_bandpass(self,x):
		x = np.asarray(x).reshape(1,-1)
		return lfilter(self.b_band,self.a_band, x, axis=-1)

	def normalize_array(self, x):
		x=np.asarray(x)
		self.mean = x.mean(axis=-1)
		self.std = x.std(axis=-1)
		return (x-self.mean)/self.std

	def normalize_one(self, xi, mu, sig, decay=-1):
		if decay<0:
			decay = self.decay
		newmu = (1.0-decay)*xi + decay*mu
		newsig = ( (1.0-decay)* (xi-mu)**2 + decay* sig**2 )**0.5
		newxi = (xi - newmu) / newsig
		return newxi, newmu, newsig

	def normalize(self, x):
		x = np.asarray(x).reshape(1,-1)
		if x.shape[-1] < self.window:
			self.window = x.shape[-1]
		x1 = self.normalize_array( x[0,0:self.window] )
		x2=[]
		for i in range(self.window , x.shape[-1]):
			newxi, self.mean,self.std=self.normalize_one(x[0,i], self.mean, self.std)
			x2.append(newxi)
		return np.append(x1,x2)

	def add_noise(self,x,nr=0.05): # add noise ratio 
		x = np.asarray(x)
		stdd=x.std(axis=-1)
		amp = abs(nr*stdd)
		return x+ np.random.normal(0.0,amp,size=x.shape)

	def drop_out(self,x,ratio=0.05): # consecutive seg drop to zero 
		x = np.asarray(x).reshape(1,-1)
		dropsz = int(x.shape[-1]*ratio)
		droppos = int(x.shape[-1]* np.random.uniform(low=0.0, high=1.0-ratio))
		x[0,droppos:droppos+dropsz]=0.0
		return x 

	def drop_zero_sparse(self,x,ratio=0.05):
		x = np.asarray(x).reshape(1,-1)
		dropsz = int(x.shape[-1]*ratio)
		while dropsz>0:
			dropsz-=1
			droppos = int(x.shape[-1]* np.random.uniform(low=0.0, high=1.0))
			x[0,max(0,min(droppos,x.shape[-1]-1))]=0.0
		return x

if __name__ == "__main__":
	a = preProcessor(125,1,45)
	x = np.random.randn(1, 200)
	print(x)
	x1 = a.filter_high(x)
	print(x1)
	print(x1.shape)
	x2 = a.normalize(x1)
	print(x2)
	print(x2.shape)