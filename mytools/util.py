#!/usr/bin/env python

import os, sys
import subprocess
import random, time
import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
import datetime

iprint = 1
iprintverb =0

class Util: 
	def __init__(self ):
		pass
	def __del__(self):
		pass

def read_lines_as_dic(fname,sep = ":"):
	res = {}
	with open(fname,"r") as fd:
		for l in fd:
			l=l.strip()
			if len(l)>0:
				st = l.split(sep)
				res[st[0].strip()]=st[1].strip()
	return res

def read_lines_as_list(fname):
	iflist = []
	with open(fname,"r") as fd:
		for l in fd:
			l=l.strip()
			if len(l)>0:
				iflist.append(l)
	return iflist

def load_key2vlist(fname, sep=" "): 
  if iprintverb: print("load lines as key and values list, dic[tarekc]=[1,2,3]")
  dic={}
  with open(fname,'r') as fd:
    for line in fd:
      st=line.strip().split(sep)
      dic[st[0].strip()]=[]
      for ee in st[1:]:
        dic[st[0].strip()].append(ee.strip())
  return dic

def filter_w(arr,ax=0,wind=3,weights=[]):
	if iprintverb: print("filter arr(#,ax) , avg wind 3, or given weights [1,1,1]")
	if len(weights)>0:
		wt=[]
		sm=float(sum(weights))
		for x in weights:
			wt.append(x/sm)
	else:
		wt=[]
		for i in range(wind):
			wt.append(1.0/wind)
	wl = len(wt)/2
	wr = len(wt)-1-wl
	res = []
	for i in range(len(arr)):
		res.append([])
		for j in range(len(arr[0])):
			if j!=ax:
				res[i].append(arr[i][j])
				continue
			sm=0.0
			sw=0.0
			for k in range(i-wl,i+1):
				if k>=0:
					sm+=arr[k][j]*wt[k-i+wl]
					sw+=wt[k-i+wl]
			for k in range(i+1,i+wr+1):
				if k<len(arr):
					sm+=arr[k][j]*wt[k-i+wl]
					sw+=wt[k-i+wl]
			res[i].append(sm/sw)
	return res

def bucket(x, low, high, dx): 
	if iprintverb: print("bucket x into bins from low to high sep dx. [low:dx:high]")
	if x<=low: return x
	nx = int((x-low)/dx)
	if nx>int((high-low)/dx): return low+int((high-low)/dx)*dx
	return nx*dx+low

def strip_letter(istr):
  ostr=[]
  for lt in istr:
    if ord(lt)>=ord('0') and ord('9')>=ord(lt):
      ostr.append(lt)
  return ''.join(ostr) 

  
if __name__ == "__main__":
	a=[[0,0.1,1],[1,0.4,1],[2,0.1,1]]
	print(a,"filter_w(a,1,3,[1,10,1])")
	print(filter_w(a,1,3,[1,10,1]))
	print("bucket(0.1,0,1,0.5)",bucket(0.1,0,1,0.5))
	print("bucket(-2.5,-3,3,0.5)",bucket(-2.5,-3,3,0.5))
	print("bucket(-2.6,-3,3,0.5)",bucket(-2.6,-3,3,0.5))


