#!/usr/bin/env python

import os, sys
import subprocess,threading
# import random, time

import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)

from hostip import get_username
from readconf import get_conf
from os.path import expanduser

iprint = 1

def invoke(cmd): # ssh into greengps and then ssh tarekc and run cmd, must use another thread.
	try:
		subprocess.call(cmd,shell=True)
	except:
		pass

if __name__ == "__main__":
	home = expanduser("~")
	mydir = sys.argv[1]
	
	thqueue = []

	fd = open('%s/hostrank.txt'%home,'r')
	for line in fd:
		host = line.split(" ")[0].strip()
		usr = get_username(host)
		# copy rsync using relative dir, like mkdir -p root dir 
		cmd = "ssh "+usr+"@"+host+" 'rsync -avz --exclude-from=/home/zhao97/syncdir/"+mydir+"/0rsync_exclude --exclude *.pyc ~/syncdir/"+mydir+" /srv/scratch/zhao97/syncdir/"+"'"
		if iprint: print(cmd)
		t=threading.Thread(target=invoke,args=(cmd,))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)
		
		# break

	for t in thqueue:
		t.join(150)