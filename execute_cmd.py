#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import sys
import getopt

"""
__author__ = 'zhijie'
__mtime__ = '2018/2/6'
"""

help_str = """execute_cmd.py -s <script-name> 被执行python脚本名称，不含扩展名 
               -m --master <master> 执行主节点
               -n --num-executors <num-executors> 计算器数目
               -e --executor-memory <executor-memory> 执行内存
               -d --driver-memory <driver-memory>   驱动内存
               -p --py-files <python files> 执行文件
    """


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hs:m:n:e:d:p:',
                                   ['script-name', 'master', 'num-executors', 'executor-memory', 'driver-memory',
                                    'py-files'])
    except getopt.GetoptError:
        print help_str
        sys.exit(2)
    params = ['yarn-cluster', '50', '2', '2', '']  # default value
    for opt_param, opt_value in opts:
        if opt_param in ('-h', '--help'):
            print help_str
            sys.exit()
        elif opt_param in ('-s', '--script-name'):
            params[4] = opt_value
        elif opt_param in ('-m', '--master'):
            params[0] = opt_value
        elif opt_param in ('-n', '--num-executors'):
            params[1] = opt_value
        elif opt_param in ('-e', '--executor-memory'):
            params[2] = opt_value
        elif opt_param in ('-d', '--driver-memory'):
            params[3] = opt_value
        elif opt_param in ('-p', '--py-files'):
            params.insert(4, opt_value)
    return tuple(params)


def run(args=tuple()):
    cmd = """source change_spark_version spark-2.1.0 && /home/bigdata/software/spark-2.1.0.7-bin-2.4.0.10/bin/spark-submit --master %s --num-executors %s --executor-memory %sg --driver-memory %sg"""
    #已有4个参数
    if len(args) == 6:
        cmd += ' --py-files %s '
    cmd += ' %s.py'
    os.system(cmd % args)#共6个参数


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run(main(sys.argv[1:]))
    else:
        print help_str
