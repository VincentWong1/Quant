#！urs/bin/usr python
# -*- coding:utf-8 -*-

"Some Tool Functions for assisting main script"

from __future__ import unicode_literals
from __future__ import division
from spark_operator import operator
import datetime
import warnings
warnings.filterwarnings('ignore')

__author__ = 'WiGi'
__mtime__ = 'June 8, 2018'

def exe_switch():
    """
    To control the script to run only on trading days
    """
    work_days = operator.filter_table_data('fdm_sor.sor_bps_working_day_config', 'work_date', date_flag = "='INTERNAL'", is_workday = "='1'")
    task_date = (datetime.date.today()+datetime.timedelta(days=-1)).strftime('%Y-%m-%d %H:%M:%S')
    if_exe = not work_days.where(work_days.work_date == task_date).rdd.isEmpty() #spark df
    return if_exe

