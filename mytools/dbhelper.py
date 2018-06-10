#!/usr/bin/env python

import os, sys
import subprocess, threading
import random, time
import platform
import inspect
mypydir =os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(mypydir)
import mysql.connector
from mysql.connector import errors
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
import datetime
from copy import deepcopy

from hostip import host2ip, ip2host, host2userip
import logging
PoolExhaustSleep = 10

DEBUG_PRINT=4
INFO_PRINT=3
WARN_PRINT=2
ERROR_PRINT=1

PRINTLEVEL=1

class Database:
	
	def __init__(self, dbname, passwd="", host=""):
		self.config = {
		  'user': 'root',
		  'password': '',
		  'host': 'localhost',
		  'database': '',
		}
		if passwd!="": self.config['password']=passwd
		if dbname!="": self.config['database']=dbname
		if host!="": self.config['host']=host
		self.pool = None
		self.connectDbPool()

	def connectDbPool(self,POOLSIZE=1): 
		try:
			if PRINTLEVEL>=INFO_PRINT: print "MySQLConnectionPool connectDbPool ..."
			cnxpool=MySQLConnectionPool(pool_name=self.config['host'],pool_size=POOLSIZE,**self.config)
			self.pool=cnxpool
			return 1
		except Exception, e:
			print "! Failed connectDbPool !"
			if PRINTLEVEL>=ERROR_PRINT: print(e)
		return -1

	def th_exec_sql(self,cmd,value,result=[],read=True,write=False,newConn=False): # no use
		cursor=None
		cnx=None
		if newConn: # exec on every node using new connection
			tmpconfig=deepcopy(self.config)
			try:
				cnx=mysql.connector.connect(**tmpconfig)
				cursor=cnx.cursor(dictionary=True)
				if read:
					cursor.execute(cmd)
					row = cursor.fetchone()# fetchone(),  fetchmany() or  fetchall()
					while row:
						result.append(row) # list append dict
						row = cursor.fetchone()
				if write:
					cursor.execute(cmd,value) # here is the difference
					cnx.commit()
				cursor.close()
				cnx.close()
			except:
				if PRINTLEVEL>=ERROR_PRINT: print("Error th_exec_sql newConn")
				pass
		else:
			done=False
			sleept=PoolExhaustSleep
			while not done:
				try:
					cnx= self.pool.get_connection()
					cursor=cnx.cursor(dictionary=True)
					if read:
						cursor.execute(cmd)
						row = cursor.fetchone()# fetchone(), fetchmany() or  fetchall()
						while row:
							result.append(row) # list append dict
							row = cursor.fetchone()
					if write:
						cursor.execute(cmd,value)
						cnx.commit()
					done=True
				except errors.PoolError as error:
					if PRINTLEVEL>=WARN_PRINT: print('pool exhausted!')
					if PRINTLEVEL>=ERROR_PRINT: print(error)
					time.sleep(sleept)
					sleept*=1.2
				except Exception, error:
					if PRINTLEVEL>=ERROR_PRINT: print('exec %s error!'%(cmd))
					if PRINTLEVEL>=ERROR_PRINT: print(error)
					done=True
				finally:
					if cursor: cursor.close()
					if cnx: cnx.close()
			
	def th_exec_raw(self,cmd,result=[],read=True,write=False,newConn=False):
		cursor=None
		cnx=None
		if newConn: # exec on every node using new connection
			tmpconfig=deepcopy(self.config)
			try:
				cnx=mysql.connector.connect(**tmpconfig)
				cursor=cnx.cursor(dictionary=True)
				cursor.execute(cmd)
				if read:
					row = cursor.fetchone()# fetchone(),  fetchmany() or  fetchall()
					while row:
						result.append(row) # list append dict
						row = cursor.fetchone()
				if write:
					cnx.commit()
				cursor.close()
				cnx.close()
			except:
				if PRINTLEVEL>=ERROR_PRINT: print("Error th_exec_raw newConn")
				pass
		else:
			done=False
			sleept=PoolExhaustSleep
			while not done:
				try:
					cnx= self.pool.get_connection()
					cursor=cnx.cursor(dictionary=True)
					cursor.execute(cmd)
					if read:
						row = cursor.fetchone()# fetchone(), fetchmany() or  fetchall()
						while row:
							result.append(row) # list append dict
							row = cursor.fetchone()
					if write:
						cnx.commit()
					done=True
				except errors.PoolError as error:
					if PRINTLEVEL>=WARN_PRINT: print('pool exhausted!')
					if PRINTLEVEL>=ERROR_PRINT: print(error)
					time.sleep(sleept)
					sleept*=1.2
				except Exception, error:
					if PRINTLEVEL>=ERROR_PRINT: print('exec %s error!'%cmd)
					if PRINTLEVEL>=ERROR_PRINT: print(error)
					done=True
				finally:
					if cursor: cursor.close()
					if cnx: cnx.close()

	def find_max_attri_val(self,tbname,attri,name=''):
		qry="SELECT MAX("+attri+") FROM "+tbname
		result=[]
		thqueue=[]
		t = threading.Thread(target=self.th_exec_raw,args=(qry,result,True,False))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)
		if PRINTLEVEL>=DEBUG_PRINT:print('find max val %s %s %s'%(name,tbname,attri))
		for th in thqueue:
			th.join()
		maxval=0
		for dic in result:
			if maxval<dic.values()[0]:
				maxval=dic.values()[0]
		return maxval

	def insert(self,tbname, paramdict):# {'attr':val, }
		qry1='INSERT INTO '+tbname+'('
		qry2=') VALUES ('
		qry3=')'
		for k,v in paramdict.items():
			qry1=qry1+k+","
			qry2=qry2+str(v)+"," 
			
		cmd = qry1.strip(",")+qry2.strip(",")+qry3
		if PRINTLEVEL>=INFO_PRINT: print(cmd)
		thqueue=[]
		t = threading.Thread(target=self.th_exec_raw,args=(cmd,[],False,True))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)
		for th in thqueue:
			th.join()

	def fetch(self,tbname, paramdict): # {'attr':[min, max], }
		qry1="SELECT "
		qry2=" FROM "+tbname+" WHERE "
		result=[]
		thqueue=[]
		for k,v in paramdict.items():
			qry1=qry1+k+","
			if len(v)>0:	# specified 
				qry2=qry2+" %s>="%k +str(v[0]) +" AND %s<="%k +str(v[1]) +" AND"
			
		cmd = qry1.strip(",")+qry2.strip("AND")
		if PRINTLEVEL>=INFO_PRINT: print(cmd)
		t = threading.Thread(target=self.th_exec_raw,args=(cmd,result,True,False))
		t.setDaemon(True)
		t.start()
		thqueue.append(t)
		for th in thqueue:
			th.join()
		return result

	def create_table(self,tbname, tabledic =None):
		tmpconfig=deepcopy(self.config)
		cnx=mysql.connector.connect(**tmpconfig)
		cursor=cnx.cursor(dictionary=True)
		cmd=gen_create_table(tbname, tabledic)
		if PRINTLEVEL>=ERROR_PRINT: print(cmd)
		cursor.execute(cmd)
		cursor.close()
		cnx.close()

	def drop_table(self,tbname):
		tmpconfig=deepcopy(self.config)
		cnx=mysql.connector.connect(**tmpconfig)
		cursor=cnx.cursor(dictionary=True)
		cursor.execute("DROP TABLE IF EXISTS "+tbname)
		cursor.close()
		cnx.close()
		if PRINTLEVEL>=INFO_PRINT: print('drop table %s'%tbname)

	def create_db(self,dbname): # don't use
		cnx= self.pool.get_connection()
		cursor=cnx.cursor(dictionary=True)
		try:
			qry="CREATE DATABASE IF NOT EXISTS "+dbname
			if PRINTLEVEL>=DEBUG_PRINT: print(qry)
			cursor.execute(qry)
		except Exception, e:
			if PRINTLEVEL>=ERROR_PRINT: print(e)
		finally:
			cursor.close()
			cnx.close()				

	def drop_db(self,dbname): # don't use
		cnx= self.pool.get_connection()
		cursor=cnx.cursor(dictionary=True)
		try:
			qry="DROP DATABASE IF EXISTS "+dbname
			if PRINTLEVEL>=DEBUG_PRINT: print(qry)
			cursor.execute(qry)
		except Exception, e:
			if PRINTLEVEL>=ERROR_PRINT: print(e)
		finally:
			cursor.close()
			cnx.close()


######################### const  ##################
TableDic={

"data":{
"PRIMARY":"sid",
"sid":["BIGINT UNSIGNED NOT NULL", 'int'],
"uid":["BIGINT UNSIGNED NOT NULL", 'int'],
"accx":["DOUBLE(16,9)",'float']	,
"accy":["DOUBLE(16,9)",'float']	,
"accz":["DOUBLE(16,9)",'float']	,
"gyrx":["DOUBLE(16,9)",'float']	,
"gyry":["DOUBLE(16,9)",'float']	,
"gyrz":["DOUBLE(16,9)",'float']	,
"gpsspeed":["DOUBLE(16,9)",'float']	,
"gpsalt":["DOUBLE(16,9)",'float']	,
"gpslng":["DOUBLE(16,9)",'float']	,
"gpslat":["DOUBLE(16,9)",'float']	,
"gpsbear":["DOUBLE(16,9)",'float']	,
"gpstime":["DOUBLE(16,9)",'float']	,
"obdspeed":["DOUBLE(16,9)",'float']	,
"obdrpm":["DOUBLE(16,9)",'float']	,
"obdmaf":["DOUBLE(16,9)",'float']	,
"obdthrot":["DOUBLE(16,9)",'float']	,
"tripid":["BIGINT UNSIGNED", 'int']	,
},
}

TABLES = {}

def gen_create_table(tbname, tabledic):
	if tabledic is None: tabledic = TableDic
	dic = tabledic[tbname]
	sql = "CREATE TABLE IF NOT EXISTS `%s` ("%tbname
	primkey = dic["PRIMARY"]
	for k,l in dic.items():
		if k!="PRIMARY":
			sql = sql+ "`%s` "%k + l[0] +","
	sql=sql+" PRIMARY KEY (%s)"%primkey+ ") ENGINE=InnoDB"
	return sql

TABLES['user'] = (
    "CREATE TABLE IF NOT EXISTS `user` ("
    "  `uid` INT UNSIGNED NOT NULL AUTO_INCREMENT,"
    "  `name` varchar(20) NOT NULL default 'default',"
    "  `password` varchar(30) NOT NULL default '123456',"
    "  `email` varchar(40) NOT NULL UNIQUE,"
    "  `token` varchar(130) UNIQUE default NULL,"
    "  `baknum` INT UNSIGNED NOT NULL default 0,"
    "  PRIMARY KEY (`uid`)"
    ") ENGINE=InnoDB")
TABLES['author'] = (
    "CREATE TABLE IF NOT EXISTS `author` ("
    "  `aid` INT UNSIGNED NOT NULL AUTO_INCREMENT,"
    "  `name` varchar(20) NOT NULL default 'default',"
    "  `numfollowers` INT UNSIGNED DEFAULT 0,"
    "  `username` varchar(20) NOT NULL default '',"
    "  `email` varchar(40) NOT NULL UNIQUE,"
    "  `baknum` INT UNSIGNED NOT NULL default 0,"
    "  PRIMARY KEY (`aid`)"
    ") ENGINE=InnoDB")
TABLES['tweets'] = (
    "CREATE TABLE IF NOT EXISTS `tweets` ("
    "  `tid` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,"
    "  `timestamp` DATETIME NOT NULL DEFAULT '2035-01-01 23:59:59',"
    "  `url` varchar(100) default '',"
    "  `content` varchar(150) not null default '',"
    "  `category` varchar(50) not null,"
    "  `author` varchar(50),"
    "  `baknum` INT UNSIGNED NOT NULL default 0,"
    "  PRIMARY KEY (`tid`)"
    ") ENGINE=InnoDB")
TABLES['prefer'] = (
    "CREATE TABLE IF NOT EXISTS `prefer` ("
    "  `uid` INT UNSIGNED NOT NULL AUTO_INCREMENT,"
    "  `category` varchar(50) not null,"
    "  `keywords` varchar(20) not null,"
    "  `weight` real default 0,"
    "  `baknum` INT UNSIGNED NOT NULL default 0,"
    "  PRIMARY KEY (`uid`,`category`,`keywords`)"
    ") ENGINE=InnoDB")

init_name2nids={
'Sports':       [ 0,10,20,30,19, 7,15,23],
'Technology':   [ 2,12,22,32, 1, 9,17,25],
'Science':      [ 4,14,24,34, 3,11,19,27],
'Entertainment':[ 5,15,25, 8, 4,12,20,28],
'News':         [ 6,11,21,31, 0, 8,16,24],

# 'health':       [ 3,13,23,33, 2,10,18,26],
# 'world':        [ 1,16,26, 9, 5,13,21,29],
# 'business':     [ 7,17,27,18, 6,14,22,30],

}

if __name__ == "__main__":
	if PRINTLEVEL>=DEBUG_PRINT: print TABLES

#cnxpool=MySQLConnectionPool(pool_name=None,pool_size=5,**kwargs)
#cnxpool.add_connection(cnx) # add existing connection to pool,raises a PoolError if the pool is full
#cnxpool.set_config(**kwargs) # no use here
#cnx= cnxpool.get_connection()#raises a PoolError if no connections are available.
# name = cnxpool.pool_name

#SELECT daytime FROM tb WHERE daytime >= NOW() or >='2014-05-18 15:00:00'
# cursor.execute("SELECT * FROM books")
#         row = cursor.fetchone()# fetchone(),  fetchmany() or  fetchall() method to fetch data from the result set.
#         while row is not None:
#			 print(row)  #printed out the content of the row and move to the next row until all rows are fetched.
#             row = cursor.fetchone()

## if row number is small:
# cursor.execute("SELECT * FROM books")
#         rows = cursor.fetchall()
#         print('Total Row(s):', cursor.rowcount)
#         for row in rows:
#             print(row)

#fetchmany() method that returns the next number of rows (n) of the result set, which allows us to balance between time and memory space.
# rows = cursor.fetchmany(size)
#         if not rows:
#             break
#         for row in rows:
#             yield row