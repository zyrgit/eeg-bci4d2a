#!/usr/bin/env python

import os, sys
import subprocess
import random, time

import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
from readconf import get_hostlist,get_allhosts

from hostip import host2ip, ip2host, host2userip, ip2tarekc, tarekc2powerport
import logging
from namehostip import get_namehostip
from servicedir import host2dir,host2servicedict
import platform
arch=platform.architecture()[0]
if arch.startswith('64'):
	if sys.platform.startswith('darwin'):
		import psutil_mac as psutil
	else:
		import psutil_linux_64 as psutil
elif arch.startswith('32'):
	import psutil_linux_32 as psutil

def get_disk_io():
	res=psutil.disk_io_counters(perdisk=False)
	ret={}
	ret['read']=res.read_bytes/1024.0/1024.0
	ret['write']=res.write_bytes/1024.0/1024.0
	return ret

def get_cpu_usage(intv=0.5,per=1): # cpu % of us, sy,  percent
	res=psutil.cpu_times_percent(intv, per>=1)
	ret={}
	if per==0:
		ret['usr']=res[0]
		ret['sys']=res[2]
		return ret
	for i in range(len(res)):
		ret[i]={}
		ret[i]['usr']=res[i][0]
		ret[i]['sys']=res[i][2]
	return ret

def get_cpu_count(hyperthread=1):
	return psutil.cpu_count(logical=(hyperthread>0))

def get_mem_usage(): # total MB, used MB
	mem=psutil.virtual_memory()
	ret={}
	ret['total']=int(mem[0]/1024/1024)
	ret['used']=int(mem[3]/1024/1024)
	return ret # total MB, used MB

def get_disk_usage(path='/',io=0):
	use=psutil.disk_usage(path)
	ret={}
	ret['total']=use[0]/1024/1024 # total MB
	ret['used']=use[3] # %
	if io==0:
		iouse=psutil.disk_io_counters(perdisk=False)
		ret['rcnt']=iouse[0] # occur times
		ret['wcnt']=iouse[1]
		ret['rmb']=iouse[2]/1024.0/1024 # volumn
		ret['wmb']=iouse[3]/1024.0/1024
		ret['rsec']=iouse[4]/1000.0 # time
		ret['wsec']=iouse[5]/1000.0
	return ret

def get_net_txrx(total=0,rlevel=1):
	if total>0: # all NIC ?
		res= psutil.net_io_counters(pernic=False)
		ret={}
		ret['tx']=res[0]/1024.0/1024.0
		ret['rx']=res[1]/1024.0/1024.0 # MB
		if rlevel>=4:
			ret['txpac']=res[2] # num of packets
			ret['rxpac']=res[3]
		if rlevel>=2:
			ret['dropin']=res[6] # num of drops
			ret['dropout']=res[7]
		if rlevel>=3:
			ret['errin']=res[4] # num of err
			ret['errout']=res[5]
		return ret
	res=psutil.net_io_counters(pernic=True)
	maxv=0
	target='' # if exception
	for nic in res.keys(): # find the correct NIC, max vol one
		if nic.startswith('eth'):
			tmp=res[nic][0]+res[nic][1]
			if tmp>maxv:
				maxv=tmp
				target=nic
	if target=='':
		return None
	ret={}
	ret['tx']=res[target][0]/1024.0/1024.0
	ret['rx']=res[target][1]/1024.0/1024.0
	if rlevel>=4:
		ret['txpac']=res[target][2]
		ret['rxpac']=res[target][3]
	if rlevel>=2:
		ret['dropin']=res[target][6]
		ret['dropout']=res[target][7]
	if rlevel>=3:
		ret['errin']=res[target][4]
		ret['errout']=res[target][5]
	return ret # [bytes_sent MB, bytes_recv MB, packets_sent, packets_recv, errin=0, errout=0, dropin=0, dropout=0]

def get_ip_addr():
	res=psutil.net_if_addrs()
	for nic in res.keys():
		for ipv in res[nic]: # v4, v6?
			if not ipv[1].startswith(':'): # addr
				if not ipv[1].startswith('127'):
					if ipv[2]!= None: # netmask none?
						ip= ipv[1]
						return ip
	return ''

def get_net_conn(raddr='',typ='tcp',summary=1):# {ip:{CONN_TIME_WAIT:1000, }, }
	if typ!='udp':
		typ='tcp4'
	else:
		typ='udp4'
	if raddr=='':
		res=psutil.net_connections(typ)
		if summary==0:
			return res
		else:
			ret={} # {ip:{CONN_TIME_WAIT:1000, }, }
			for con in res:
				if len(con[4])==0:
					continue
				rip=con[4][0]
				status=con[5]
				if not rip in ret.keys():
					ret[rip]={status:1}
				elif not status in ret[rip].keys():
					ret[rip][status]=1
				else:
					ret[rip][status]+=1
			return ret
	else:# specify remote ip
		res=psutil.net_connections(typ)
		ret={} # {ip:{CONN_TIME_WAIT:1000, }, }
		for con in res:
			if len(con[4])==0:
				continue
			rip=con[4][0]
			if rip==raddr:
				status=con[5]
				if not rip in ret.keys():
					ret[rip]={status:1}
				elif not status in ret[rip].keys():
					ret[rip][status]=1
				else:
					ret[rip][status]+=1
		return ret

def get_procnum(name=''):
	ret={}
	if name=='':
		for proc in psutil.process_iter():
			dic=proc.as_dict(attrs=['pid', 'name'])
			if not dic['name'] in ret.keys():
				ret[dic['name']]=1
			else:
				ret[dic['name']]+=1
		return ret
	ret={name:0}
	for proc in psutil.process_iter():
		if name==proc.name():
			ret[name]+=1
	return ret
	
def get_power(hoststr='',suffix='',LOG=0):
	if not hoststr:
		hoststr=get_ip_addr()
	try:
		if hoststr.startswith('tarekc'):
			name=hoststr.split('.')[0]
		elif hoststr[0].isalpha(): # t1 
			hip=host2ip[hoststr]
			if hip in ip2tarekc.keys(): 
				name=ip2tarekc[hip]
			else:
				name=get_namehostip('hname',hip) # help from namehostip
		elif hoststr.startswith('1'): # 172.
			if hoststr in ip2tarekc.keys(): 
				name=ip2tarekc[hoststr] # is ip already in
			else:
				name=get_namehostip('hname',hoststr) # help from namehostip
		
		pport=tarekc2powerport[name]
		#print 'Power for '+hoststr+' '+pport
		cmd='php '+mypydir+'/avocent_measure.php '+pport
		proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,stderr=open('/dev/null', 'w'))
		output = proc.communicate()[0].strip()
		
		if LOG:
			fid=open(host2dir(name)+'/power'+suffix+'.log','a')
			fid.write(('%.4f'%time.time())+' '+hoststr+' '+output+'\n')
			fid.close()
			# SET LOG
			# logging.basicConfig(level=logging.INFO)
			# logger = logging.getLogger(__name__)
			# handler = logging.FileHandler(host2dir(name)+'/power'+suffix+'.log')
			# handler.setLevel(logging.INFO)
			# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
			# handler.setFormatter(formatter)
			# logger.addHandler(handler)
			#logger.info(hoststr+' '+output)
	except:
		print hoststr+' '+name+' has NO power measure!'
		output=''
	return output

if __name__ == "__main__":

	pcpu=1
	
	#print psutil.cpu_times(percpu=(pcpu==1))#  cpu user=37202.95, nice=0.0, system=25829.3, idle=577675.09
	#print psutil.cpu_percent(interval=1, percpu=(pcpu==1))
	#print psutil.cpu_times_percent(interval=1, percpu=(pcpu==1))#[scputimes(user=4.0, nice=0.0, system=4.0, idle=91.9), scputimes(user=1.0, nice=0.0, system=0.0, idle=99.0), scputimes(user=5.0, nice=0.0, system=2.0, idle=93.0), scputimes(user=2.0, nice=0.0, system=1.0, idle=97.0)]
	#print psutil.cpu_times(percpu=(pcpu==1))#  cpu user=37202.95, nice=0.0, system=25829.3, idle=577675.09
	print 'get_ip_addr:'
	print get_ip_addr()
	print 'get_cpu_usage:'
	print get_cpu_usage()
	print 'get_cpu_count:'
	print get_cpu_count()
	print 'get_mem_usage:'
	print get_mem_usage()
	print 'get_net_conn:'
	print get_net_conn()
	print 'get_net_txrx:'
	print get_net_txrx(total=0,rlevel=4)
	print 'get_disk_usage:'
	print get_disk_usage()
	print 'get_disk_io()'
	print get_disk_io()
	#print 'get_power:'
	#print get_power()


