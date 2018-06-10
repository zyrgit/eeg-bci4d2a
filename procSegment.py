#!/usr/bin/env python

import os, sys
# import subprocess
# import random, time
import inspect
# import collections
# import math
# from shutil import copy2, move as movefile
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
# sys.path.append(mypydir+"/mytools")

import numpy as np
import pandas as pd

from fft import FFT
from preprocess import preProcessor

iprint =1

if_notch_filter = 0
if_normalize = 0

class procSegment: # files only contain used channels, not all raw...
	def __init__(self,numFeatures,shiftRatio,sampleRate, lowcut, highcut, secondPerFrame, params):
		self.nFeatures = numFeatures # 22 or 25 channels
		self.shiftRatio = shiftRatio # shift window by this ratio. not used.
		self.sampleRate = sampleRate # 125 Hz
		self.lowcut = lowcut
		self.highcut = highcut
		self.secondPerFrame = secondPerFrame
		if "augmentTimes" in params.keys(): self.augmentTimes = params["augmentTimes"]
		if "init_window_size" in params.keys(): self.init_window_size = params["init_window_size"]
		if "stateStep" in params.keys(): self.stateStep = params["stateStep"]

		if "convertToSampleRate" in params.keys(): 
			self.convertToSampleRate = params["convertToSampleRate"]
			if self.convertToSampleRate <0 :
				self.convertToSampleRate = self.sampleRate
		else:
			self.convertToSampleRate = self.sampleRate
		self.nfft=-1


	def augment_file(self,fname): # per trial file, change sp-rate, dropout and add noise
		data = pd.read_csv(fname)# include header !
		data.interpolate(method='linear', axis=0, inplace=True) # inplace=1 otherwise needs return val.
		data.fillna(0, inplace=True)
		nRow=data.shape[0]
		nCol=data.shape[1]
		pp = preProcessor(self.sampleRate,self.lowcut,self.highcut,weight_decay=0.999,init_window_size=self.init_window_size)

		if self.sampleRate!=self.convertToSampleRate:
			newrow = int( float(nRow) / self.sampleRate * self.convertToSampleRate)
			newda = pd.DataFrame(0,index=np.arange(newrow),columns=data.columns)

		for c in range(0,self.nFeatures):
			x=np.asarray(data.iloc[: , c])
			if if_notch_filter:
				x = pp.filter_not( x )
			x = pp.filter_bandpass( x )
			if if_normalize:
				x = pp.normalize(x)
			if x.shape[0]==1:
				x=x[0]
			if self.sampleRate!=self.convertToSampleRate:
				x1=pp.change_sample_rate(x,self.sampleRate,self.convertToSampleRate)
				newda.iloc[:,c] = x1
			else:
				data.iloc[: , c] = x
			# print(x)
			# sys.exit(0)
		if self.sampleRate!=self.convertToSampleRate:
			newda.iloc[:,self.nFeatures] = data.iloc[0 , self.nFeatures] # cp label. 
			data=newda
			nRow=data.shape[0]

		augtime = []
		auglabel = []
		# original signal first
		ind = 0
		dtime = []
		dlabel = []
		SampPerFrame = int(self.convertToSampleRate * self.secondPerFrame)

		while ind+SampPerFrame <=nRow:
			timeStep = []
			label=-1
			for c in range(0,self.nFeatures):
				x=np.asarray(data.iloc[ind:ind+SampPerFrame , c]) # make copy
				timeStep.append(x)
				label=(data.iloc[ind,self.nFeatures])

			ind+=int(SampPerFrame * self.shiftRatio)
			dtime.append(timeStep)
			dlabel.append(label)
		
		if not (len(dtime)%self.stateStep==0):
			print(__file__.split("/")[-1],"bad len?",fname,len(dtime),nRow)

		augtime.append(dtime) # remain format of steps.
		auglabel.append(dlabel)
		# augment # times, increase to #+1 size: 
		for i in range(self.augmentTimes): 
			ind = 0
			dtime = []
			dlabel = []
			while ind+SampPerFrame <=nRow:
				timeStep = []
				label=-1
				for c in range(0,self.nFeatures):
					x=np.asarray(data.iloc[ind:ind+SampPerFrame , c]) # make copy
					x1=pp.add_noise(x,nr=0.01)
					x1=pp.drop_zero_sparse(x1,ratio=0.1) # reshaped(1,-1)
					timeStep.append(x1[0])
					label=(data.iloc[ind,self.nFeatures])

				ind+=int(SampPerFrame * self.shiftRatio)
				dtime.append(timeStep)
				dlabel.append(label)

			augtime.append(dtime) # remain format of steps.
			auglabel.append(dlabel)

		return augtime,auglabel # [[ori],[aug],,],,

	def augment_file_time_freq(self,fname): # per trial 3s, dropout and add noise
		data = pd.read_csv(fname)# include header !
		data.interpolate(method='linear', axis=0, inplace=True) # inplace=1 otherwise needs return val.
		data.fillna(0, inplace=True)
		nRow=data.shape[0]
		nCol=data.shape[1]
		pp = preProcessor(self.sampleRate, self.lowcut ,self.highcut,weight_decay=0.999, init_window_size=self.init_window_size)
		filt = FFT(N=self.nfft, sampleRate=self.sampleRate)
		for c in range(0,self.nFeatures):
			x=np.asarray(data.iloc[: , c])
			if if_notch_filter:
				x = pp.filter_not( x )
			x = pp.filter_high( x )
			if if_normalize:
				x = pp.normalize(x)
			if x.shape[0]==1:
				x=x[0]
			bad=-1
			if np.any(np.isnan(x)):
				bad=1
			if bad==1:
				print("nan! when convert_file")
				print(data.iloc[: , c])
				sys.exit(0)
			data.iloc[: , c] = x
			# print(x)
			# sys.exit(0)
		augtime = []
		augfreq = []
		auglabel = []
		# original signal first
		ind = 0
		dtime = []
		dfreq = []
		dlabel = []
		SampPerFrame = int(self.sampleRate * self.secondPerFrame)

		while ind+SampPerFrame <=nRow:
			timeStep = []
			freqStep = []
			label=-1
			for c in range(0,self.nFeatures):
				x=np.asarray(data.iloc[ind:ind+SampPerFrame , c]) # make copy
				timeStep.append(x)
				if if_sum_bands:
					filt.N_fft( x )
					y=filt.sum_freq_band(bands)
				else:
					y=filt.get_rfft_without_DC(x)
				freqStep.append(y)
				label=(data.iloc[ind,self.nFeatures])

			ind+=int(SampPerFrame * self.shiftRatio)
			dtime.append(timeStep)
			dfreq.append(freqStep)
			dlabel.append(label)
		
		if not (len(dtime)%self.stateStep==0):
			print(__file__.split("/")[-1],"bad len?",fname,len(dtime),nRow)
		# assert(len(dtime)%self.stateStep==0)
		augtime.append(dtime) # remain format of steps.
		augfreq.append(dfreq)
		auglabel.append(dlabel)
		# augment # times, increase to #+1 size: 
		for i in range(self.augmentTimes): 
			ind = 0
			dtime = []
			dfreq = []
			dlabel = []
			while ind+SampPerFrame <=nRow:
				timeStep = []
				freqStep = []
				label=-1
				for c in range(0,self.nFeatures):
					x=np.asarray(data.iloc[ind:ind+SampPerFrame , c]) # make copy

					x1=pp.add_noise(x,nr=0.01)
					x1=pp.drop_zero_sparse(x1,ratio=0.1) # reshaped(1,-1)
					timeStep.append(x1[0])
					
					x2 = pp.drop_zero_sparse(x,ratio=0.1)
					if if_sum_bands:
						filt.N_fft( x2 )
						y=filt.sum_freq_band(bands)
					else:
						y=filt.get_rfft_without_DC(x2)
					freqStep.append(y)
					label=(data.iloc[ind,self.nFeatures])

				ind+=int(SampPerFrame * self.shiftRatio)
				dtime.append(timeStep)
				dfreq.append(freqStep)
				dlabel.append(label)

			# assert(len(dtime)%self.stateStep==0)
			augtime.append(dtime) # remain format of steps.
			augfreq.append(dfreq)
			auglabel.append(dlabel)

		return augtime,augfreq,auglabel # [[ori],[aug],,],,
	

if __name__ == "__main__":
	a = procSegment(25,128,0.207208,125,0.5,62,0.888)
	x1,x2,lb= a.augment_file("data22/data-750-0-1.txt")
	print(x1)
	x1=np.asarray(x1)
	print(x1.shape)