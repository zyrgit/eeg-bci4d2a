#!/usr/bin/env python

import os, sys
import subprocess
import random
import socket

import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)

from hostip import host2ip, ip2host, host2userip, tarekc2powerport, ip2tarekc

def get_host_name_short():
	return socket.gethostname().split(".")[0]

def get_my_ip():
	return ([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])

def get_namehostip(targettype,who):
	try:
		if who.startswith('tarekc'):
			inputtype='hname'
			name=who.split('.')[0]

		elif who[0].isalpha(): # t1 
			inputtype='tname' # must put before exception
			name=''
			hip=''
			hip=host2ip[who]
			name=ip2tarekc[hip]

		elif who.startswith('1'): # 172.
			inputtype='ip' # must put before exception
			name=''
			name=ip2tarekc[who] 

		else:
			print 'namehostip.py: '+who+' ????? t1? tarekc? ip?'
			return ''
	except:
		print 'namehostip.py EXCEPTION translating '+who+' from '+inputtype+' to '+targettype
	
	if targettype=='hname': # e.g. return 'tarekc01'
		if inputtype=='hname': # e.g. input 'tarekc01'
			return name
		elif inputtype=='tname':# e.g. input 't11'
			if name!='':
				return name
			else:
				if hip!='':
					try:
						name=socket.gethostbyaddr(hip)[0].split('.')[0]
						return name
					except:
						print 'namehostip.py unknown ip: '+hip
						return ''
				else:
					print 'Please add '+who+' to /etc/hosts, hostip.py: host2ip !'
					return ''
		elif inputtype=='ip':# e.g. input '172.22.68.89'
			if name!='':
				return name
			else:
				try:
					name=socket.gethostbyaddr(who)[0].split('.')[0]
					return name
				except:
					print 'namehostip.py unknown ip: '+who
					return ''

	elif targettype=='tname': # e.g. return 't11'
		if inputtype=='hname':# e.g. input 'tarekc01'
			ip=''
			for k in ip2tarekc.keys():
				if ip2tarekc[k]==name:
					ip= k
					break
			if ip!='':
				try:
					return ip2host[ip]
				except:
					print 'Please add '+ip+' '+who+' to /etc/hosts, hostip.py: ip2host !'
					return ''
			else:
				fullname=name+'.cs.illinois.edu'
				try:
					ip=socket.gethostbyname(fullname)
				except:
					print 'namehostip.py unknown host: '+fullname
					return ''
				try:
					return ip2host[ip]
				except:
					print 'Please add '+ip+' '+fullname+' to /etc/hosts, hostip.py: ip2host !'
					return ''

		elif inputtype=='tname':# e.g. input 't11'
			return who

		elif inputtype=='ip':# e.g. input '172.22.68.89'
			try:
				return ip2host[who]
			except:
				print 'Please add '+who+' to /etc/hosts, hostip.py: ip2host !'
				return ''

	elif targettype=='ip': # e.g. return '128.174.236.10'
		if inputtype=='hname':# e.g. input 'tarekc01'
			ip=''
			for k in ip2tarekc.keys():
				if ip2tarekc[k]==name:
					ip= k
					break
			if ip!='':
				return ip
			else:
				fullname=name+'.cs.illinois.edu'
				try:
					ip=socket.gethostbyname(fullname)
					return ip
				except:
					print 'namehostip.py unknown host: '+fullname
					return ''
				
		elif inputtype=='tname':# e.g. input 't11'
			if hip!='':
				return hip
			else:
				print 'Please add '+who+' to /etc/hosts, hostip.py: host2ip !'
				return ''

		elif inputtype=='ip':# e.g. input '172.22.68.89'
			return who
	else:
		print 'namehostip.py targettype unknown!'
		return ''

if __name__ == "__main__":
	if len(sys.argv)==2:
		ip=sys.argv[1]
		if ip[0].isalpha():
			ip=host2ip[ip]

		name=socket.gethostbyaddr(ip)[0].split('.')[0]
		port = tarekc2powerport[name]
		print port+' '+name

	if len(sys.argv)==3:
		typ=sys.argv[1]
		who=sys.argv[2]
		res=get_namehostip(typ,who)
		print who+' :=> '+res

	else:

		ftb=open('namehostip.txt','w')
		for hip in ip2host.keys():
			try:
				name=socket.gethostbyaddr(hip)[0].split('.')[0]
				ftb.write("'"+hip+"'"+': '+"'"+name+"',"+'\n')
			except:
				print '! unable to get hostname of '+hip
				#ftb.write("'"+hip+"'"+': '+"'',"+'\n')
		ftb.close()
	
	#cmd='php '+mypydir+'/avocent_measure.php '+port
	#proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
	#output = proc.communicate()[0]
