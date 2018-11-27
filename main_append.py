#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Main exector Script"

from __future__ import unicode_literals
from Conn_Hive import initial_data
from Cal_DirctProb import cal_dirct_prob
import pandas as pd
from dateutil.parser import parse
import datetime
from Potential_Trans_Hive import Insert_Table, Add_UUID
from Controller_Tool import exe_switch
from spark_operator import operator, spark
import warnings
warnings.filterwarnings('ignore')

__author__ = "WiGi"
__mtime__ = "11 May 2018"

if exe_switch():
    stat_all = []
    accum_ret_all = pd.DataFrame(columns = ['fund_code', 'holding_period', 'start_date', 'accum_ret'])
    stat_all_df = pd.DataFrame()
    Input_T1 = pd.DataFrame()
    Input_T2 = pd.DataFrame()

    dataset_all, fee_all, rank = initial_data()
    rank.reset_index(inplace=True)
    rank['index'] = rank['index'] + 1
    rank.rename(columns={'index': 'rank_key', 'id': 'Fk'}, inplace=True)
    """
    j = 0
    """
    data_slice = dataset_all.groupby('fundcode')
    for fundcode, data_group in data_slice:
        """
        if j<=1:
        """
        accum_ret_df = []
        stat = []; stat_with_fee = []
        print fundcode
        dataset = data_group.copy()
        # dataset.TradingDay = pd.to_datetime(dataset.TradingDay, format='%Y/%m/%d %H:%M:%S')
        dataset.TradingDay = dataset.TradingDay.apply(lambda x: parse(x))  # str -> datetime
        dataset = dataset.dropna(axis=0, how='any')
        dataset.loc[:, ['nvr', 'nvr_growthrate']] = dataset.loc[:, ['nvr', 'nvr_growthrate']].apply(lambda x: x.astype('float'), axis=1)
        dataset.set_index('TradingDay', inplace=True)
        dataset.sort_index(inplace=True)
        """
            Remove Closed Time
        """
        for i in range(len(dataset)):
            if (dataset.index[i + 1] - dataset.index[i]).days == 1:
                break
        dataset = dataset[i:]
        """
            Set Parameters
        """
        fee_record = fee_all[fee_all.fundcode == fundcode]
        if not fee_record.empty:
            apply_fee = fee_record.apply_fee.astype('float') if not pd.isnull(fee_record.apply_fee).values else 0.0
            redeem_fee = fee_record.redeem_fee.astype('float') if not pd.isnull(fee_record.redeem_fee).values else 0.0
        else:
            apply_fee = 0.0; redeem_fee = 0.0
        Cal_Dirct_Prob = cal_dirct_prob(dataset.index, dataset.nvr, fundcode, apply_fee, redeem_fee)
        #accum_ret = Cal_Dirct_Prob.Calculate_Accumret(False)
        #accum_ret_all.extend(accum_ret)
        accum_ret_df = Cal_Dirct_Prob.Classify_Allret_with_Fee(False) # df  'trade_date'
        accum_ret_all = accum_ret_all.append(accum_ret_df[['fund_code', 'holding_period', 'start_date', 'accum_ret']].loc[accum_ret_df.index == accum_ret_df.index.max()])
        for dirct_prob_window in [1, 3, 5]:
            stat = Cal_Dirct_Prob.Direct_Prob_Add(dirct_prob_window, False)  # list
            stat_with_fee = Cal_Dirct_Prob.Direct_Prob_Add(dirct_prob_window, True)  # list
            stat_all.extend(stat); stat_all.extend(stat_with_fee)
        """
        else:
            pass
        j += 1
        """

    # Tabel 1
    accum_ret_all.reset_index(inplace=True)
    accum_ret_all["index"] = accum_ret_all["index"].apply(lambda x: x.date())
    accum_ret_all = Add_UUID(accum_ret_all)
    accum_ret_all["updatetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")#datetime.date.today().strftime("%Y-%m-%d")
    Input_T1 = accum_ret_all[['Id', 'fund_code', 'accum_ret', 'holding_period', 'start_date', 'index','updatetime']]
    print Input_T1.head()
    history_data1 = spark.table('fbidm.fnd_fund_hd_fxd_prid_return_standby').select(['EndDate']).distinct().toPandas()
    print history_data1.head()
    if history_data1['EndDate'].max() != (datetime.date.today() + datetime.timedelta(days=-1)): #and not history_data1.empty:
        Insert_Table(Input_T1, 'fbidm.fnd_fund_hd_fxd_prid_return_standby')

    # Tabel 2
    stat_all_df = pd.DataFrame(stat_all, columns=[u'fund_code', u'end_date', u'holding_period',
                                                  u'start_date', u'total_num', u'period', u'withFee',
                                                  u'sample_num', u'probability', u'rank_key'])
    stat_all_df = stat_all_df.merge(rank, on='rank_key', how='left')
    stat_all_df = Add_UUID(stat_all_df)
    stat_all_df["updatetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Input_T2 = stat_all_df[['Id', 'fund_code', 'start_date', 'end_date', 'Fk', 'sample_num', 'total_num',
                            'probability', 'holding_period', 'period', 'withFee', 'updatetime']]
    print Input_T2.head()
    history_data2 = spark.table('fbidm.fnd_fund_future_potential_standby').select(['EndDate']).distinct().toPandas()
    print history_data2.head()
    if history_data2['EndDate'].max() != (datetime.date.today() + datetime.timedelta(days=-1)): #and not history_data2.empty:
        Insert_Table(Input_T2, 'fbidm.fnd_fund_future_potential_standby')