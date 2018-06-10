#!/usr/bin/env python

import os, sys
import subprocess, threading
# import random, time
import operator

import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
from os.path import expanduser

from hostip import ip2tarekc, get_username
from readconf import get_conf
from namehostip import get_my_ip

iprint = 1
h2score = {}
h2mem = {}
h2cpu = {}

MYIP=get_my_ip()

def invoke_mem(cmd,host): 
	global h2mem
	h2mem[host]=0
	try:
		output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0]
		h2mem[host]= int(output.split(":")[1].strip().split(' ')[0])
		if iprint: print(host+" "+str(h2mem[host]))
	except:
		if iprint: print(host+" die")
		pass

def invoke_cpu(cmd,host): 
	global h2cpu
	h2cpu[host]=0
	try:
		output = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0]
		h2cpu[host]= int(output.strip())
		if iprint: print(host+" "+str(h2cpu[host]))
	except:
		if iprint: print(host+" die")
		pass

if __name__ == "__main__":
	home = expanduser("~") # runs on greengps or tarekc, /home/zhao97

	dirpath=os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))+"/..")

	cnt= 0
	thqueue = []

	for host in ip2tarekc.values():
		usr = get_username(host)
		#ssh zhao97@tarekc61 cat /proc/meminfo | grep MemFree
		cmd = 'ssh '+usr+'@'+host+" cat /proc/meminfo | grep MemFree"
		if iprint:
			print(cmd)
		t=threading.Thread(target=invoke_mem,args=(cmd,host))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)

		#ssh zhao97@tarekc61 nproc
		cmd = 'ssh '+usr+'@'+host+" nproc"
		if iprint:
			print(cmd)
		t=threading.Thread(target=invoke_cpu,args=(cmd,host))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)

	for t in thqueue:
		t.join(5)

	for host in ip2tarekc.values():
		try:
			score = h2mem[host]*h2cpu[host]
			if score>0:
				h2score[host] = score
		except:
			pass

	# dead?
	try:
		fd= open("%s/0deadhosts"%(home),'r')
		for h in fd:
			h2score.pop(h.strip(),None)
			print("- In ~/0deadhosts: "+h)
		fd.close()
	except:
		print("No ~/0deadhosts")

	# sort and print
	sorted_x = sorted(h2score.items(), key=operator.itemgetter(1), reverse=True)
	if iprint:print(sorted_x)
	print(">> Healthy total available hosts: "+str(len(sorted_x)))

	fd= open("%s/hostrank.txt"%home,'w')
	for tup in sorted_x:
		fd.write(tup[0]+" "+str(tup[1])+" "+str(h2cpu[tup[0]])+"\n")
	fd.close()

	fd= open("%s/hostnum.txt"%home,'w')
	fd.write(str(len(sorted_x)))
	fd.close()

	if get_username(MYIP)!="zhao97":
		cmd = "scp %s/hostnum.txt "%(home)+" zhao97@tarekc55:/home/zhao97/"
		if iprint: print(cmd)
		subprocess.call(cmd,shell=True)
		cmd = "scp %s/hostrank.txt "%(home)+" zhao97@tarekc55:/home/zhao97/"
		if iprint: print(cmd)
		subprocess.call(cmd,shell=True)

	if get_username(MYIP)!="yiran":
		cmd = "scp %s/hostnum.txt "%(home)+" yiran@128.174.236.189:/home/yiran/"
		if iprint: print(cmd)
		subprocess.call(cmd,shell=True)
		cmd = "scp %s/hostrank.txt "%(home)+" yiran@128.174.236.189:/home/yiran/"
		if iprint: print(cmd)
		subprocess.call(cmd,shell=True)
	