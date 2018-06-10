#!/bin/bash
myip=$(hostname -I)

mydir=$(pwd)
echo "cp $mydir to tarekc home syncdir..."

mails=$(echo $mydir | tr "/" "\n")

for addr in $mails
do
    thisdir=$addr
done

echo rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' --rsync-path="mkdir -p /home/zhao97/syncdir && rsync"  $mydir zhao97@tarekc55:/home/zhao97/syncdir/

rsync -avz --progress --exclude '*.pyc' --exclude-from '0rsync_exclude' --rsync-path="mkdir -p /home/zhao97/syncdir && rsync"  $mydir zhao97@tarekc55:/home/zhao97/syncdir/

nocp2srv=$1

if [ -z "$nocp2srv" ]; then # no arg, then copy to srv
	echo ssh zhao97@tarekc55 "python ~/syncdir/$thisdir/mytools/cpthome2srv_t.py $thisdir"
ssh zhao97@tarekc55 "python ~/syncdir/$thisdir/mytools/cpthome2srv_t.py $thisdir"

echo "copy to srv done!"

else

echo "copy to home done!"

fi