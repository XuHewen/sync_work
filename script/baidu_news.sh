#!/bin/sh
ps -ef | grep baidu_news.py | grep -v grep
if [ $? -ne 0 ]
then
    echo "start process ..."
    cd /home/xuhewen/work/yff
    /home/xuhewen/miniconda3/bin/python baidu_news.py
else
    echo "process is still runing....."
fi