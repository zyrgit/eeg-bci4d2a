#!/usr/bin/env python

import os, sys
# import subprocess
# import random, time

import inspect
# mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
# sys.path.append(mypydir)

# from hostip import host2ip, ip2host, host2userip
CUT='='

def get_conf(fpath,typ, firstbreak = True, delimiter=CUT):
	res=''
	try:
		if not ( fpath.startswith('/') ):
			abspath=os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
			fpath = abspath+"/../"+ fpath
			#print(fpath)

		conf=open(fpath,'r')
		for line in conf:
			if line.split(delimiter,1)[0].strip()==typ: 
				res=line.split(delimiter,1)[1].split('#')[0].strip()
				if firstbreak:
					break # last one wins
		conf.close()
	except:
		print(fpath)
	return res

def get_conf_int(fpath,typ):
	return int(get_conf(fpath,typ))

def get_conf_float(fpath,typ):
	return float(get_conf(fpath,typ))

def get_conf_str(fpath,typ):
	return (get_conf(fpath,typ))

def get_list_startswith(fpath,typ, delimiter=" "): # tarekc =1 2 3 4
	res=[]
	tmp = get_conf(fpath,typ)
	if tmp=="":
		return res
	res=tmp.split(delimiter)
	return [r.strip() for r in res]

def get_dic_startswith(fpath,entryName, delimiter=" "): # tarekc = fft:256 epc:10
	res={}
	tmp = get_conf(fpath,entryName)
	if tmp=="":
		return res
	st=tmp.split(delimiter)
	for n in st:
		kv = n.split(":")
		res[kv[0].strip()]=kv[1].strip()
	return res

if __name__ == "__main__":
	# LOAD CONFIG at parent dir
	#dirname=os.path.dirname(sys.argv[0])
	#	pathname=os.path.abspath(dirname)
	currentframe= inspect.getfile(inspect.currentframe()) # script filename
	print 'currentframe:'+currentframe
	abspath=os.path.abspath(os.path.dirname(currentframe))
	print 'abspath:'+abspath

