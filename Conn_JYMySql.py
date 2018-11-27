#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Connect MySQL of Juyuan and extract all fund data and benchmark data, moreover transform it to pandas.DataFrame from spark.dataframe"

from __future__ import unicode_literals
import MySQLdb
import pandas as pd
import tushare as ts
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

__author__ = "WiGi"
__mtime__ = "10 May 2018"

class extract_data(object):
    def __init__(self, host="10.37.89.134", user="HSFDSSusr", pswd="pm5Q4kCc40ep", db="hsfdssdb"):
        self.host = host
        self.user = user
        self.pswd = pswd
        self.db = db

    def conn_mysql(self):
        db = MySQLdb.connect(self.host, self.user, self.pswd, self.db)
        return db.cursor()

    def extract_fmysql(self, cursor, sql_command):
        cursor.execute(sql_command)
        return cursor.fetchall()

    def initial_data(self):
        sql_cmd1 = """select t2.SecuCode as fundcode
              ,t3.TradingDay as TradingDay
              ,t3.UnitNVRestored as nvr
              ,t3.NVRDailyGrowthRate as nvr_growthrate
        from mf_fundarchives t1
          join secumain t2
          on t1.InnerCode = t2.InnerCode
          join mf_fundnetvaluere t3
          on t1.InnerCode = t3.InnerCode
        where (t1.fundtypecode = 1101 and t1.type in (2,3,6,7,9) and t1.investmenttype <> 7) and 
              (t1.ExpireDate > curdate() or t1.ExpireDate is null) and 
              t1.StartDate <= date_add(curdate(),interval -1 year)
        """

        sql_cmd2 = """select t2.SecuCode as secucode 
            ,t1.TradingDay as TradingDay
            ,t1.IndexValue as IndexValue
            ,t1.ValueDailyGrowthRate as ValueDailyGrowthRate
            from MF_IndexReturnHis t1
            join
            secumain t2
            on t1.indexcode= t2.innercode
            where t2.secucode = '000300' and t2.SecuCategory = 4
        """

        sql_cmd3 = """
                    select t2.secucode as secucode
                          ,t1.canceldate as canceldate
                          ,t1.maxchargerate as maxchargerate
                          ,t1.chargeratetype as chargeratetype
                    from mf_chargeratenew t1
                    join secumain t2
                    on t1.InnerCode = t2.InnerCode
                    join mf_fundarchives t3
                    on t2.InnerCode
                    = t3.InnerCode
                    where t1.chargeratetype in (12000,12200,11010,11210) and t1.chargerateunit=6 and
                         (t3.fundtypecode = 1101 and t3.type = 2 and t3.investmenttype <> 7) and 
                         (t3.ExpireDate > curdate() or t3.ExpireDate is null) and 
                          t3.StartDate <= date_add(curdate(),interval -1 year)
        """

        #基金数据
        cursor = self.conn_mysql()
        df = self.extract_fmysql(cursor, sql_cmd1)
        dataset = pd.DataFrame(list(df), columns = ["fundcode", "TradingDay", "nvr", "nvr_growthrate"]) #261支基金

        #沪深300市场行情数据
        df_bm = self.extract_fmysql(cursor, sql_cmd2)
        dataset_bm = pd.DataFrame(list(df_bm), columns = ["SecuCode", "TradingDay", "IndexValue", "ValueDailyGrowthRate"]).fillna(value = 0) #沪深300指数
        dataset_bm = dataset_bm.sort_values(by='TradingDay').set_index('TradingDay')

        #基金费率数据
        df_fee = self.extract_fmysql(cursor, sql_cmd3)
        dataset_fee = pd.DataFrame(list(df_fee), columns=['SecuCode', 'canceldate', 'maxchargerate', 'chargeratetype']).fillna(value=np.nan)
        dataset_fee.loc[pd.isnull(dataset_fee.canceldate), 'canceldate'] = datetime.datetime(2099, 1, 1)
        dataset_fee_slices = dataset_fee.groupby("SecuCode")
        fee = []
        for fundcode, data_fee_group in dataset_fee_slices:
            redeem_fee_group = data_fee_group.loc[data_fee_group.chargeratetype.isin([12000, 12200])]
            redeem_discount_default = 1.0
            if len(redeem_fee_group) != 0:
                redeem_fee = float(redeem_fee_group.sort_values(['canceldate', 'maxchargerate'])['maxchargerate'].iloc[-1]) * redeem_discount_default * 0.01
            else:
                redeem_fee = 0.0

            apply_fee_group = data_fee_group.loc[data_fee_group.chargeratetype.isin([11210, 11010])]
            apply_discount_default = 0.1
            if len(apply_fee_group) != 0:
                apply_fee = float(apply_fee_group.sort_values(['canceldate', 'maxchargerate'])['maxchargerate'].iloc[-1]) * apply_discount_default * 0.01
            else:
                apply_fee = 0.0

            fund_fee = [fundcode, apply_fee, apply_discount_default, redeem_fee, redeem_discount_default]
            fee.append(fund_fee)
        fee = pd.DataFrame(fee, columns=['fundcode', 'apply_fee', 'apply_discount_default', 'redeem_fee','redeem_discount_default'])

        #无风险利率：一年期存款利率
        deposit_rate = ts.get_deposit_rate().loc[ts.get_deposit_rate()['deposit_type'] == u'定期存款整存整取(一年)']
        deposit_rate['deposit_rate_1y'] = deposit_rate['rate'].astype('float') * 0.01
        # deposit_rate.rename(columns = {'rate':'deposit_rate_1y'}, inplace = True)
        deposit_rate = deposit_rate.set_index('date').sort_index()
        deposit_rate.index = pd.to_datetime(deposit_rate.index, format='%Y/%m/%d')
        deposit_rate.drop(['deposit_type', 'rate'], axis=1, inplace=True)

        return  dataset, dataset_bm, fee, deposit_rate

if __name__ == '__main__':
    ExtractData = extract_data()
    dataset, dataset_bm, fee, deposit_rate = ExtractData.initial_data()
    pass
