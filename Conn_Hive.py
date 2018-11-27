#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Connect MySQL of Juyuan and extract all fund data and benchmark data, moreover transform it to pandas.DataFrame from spark.dataframe"

from __future__ import unicode_literals
from spark_operator import operator
import datetime
import warnings
warnings.filterwarnings('ignore')

__author__ = "WiGi"
__mtime__ = "10 May 2018"

def initial_data():

    sql_cmd1 = """select t1.SecurityCode as fundcode
                        ,t2.TradingDay as TradingDay
                        ,t2.UnitNVRestored as nvr
                        ,t2.NVRDailyGrowthRate as nvr_growthrate
                        ,t1.StartDate as Startdate
                  from fdm_sor.sor_evt_mf_fundarchives t1
                  join fdm_sor.sor_prd_mf_fundnetvaluere_d t2
                  on t1.InnerCode=t2.InnerCode
                  where (t1.fundtypecode=1101 and t1.type in (2,3,6,7,9) and t1.investmenttype <> 7) and
                        (t1.ExpireDate>from_unixtime(unix_timestamp(),'yyyy-MM-dd') or t1.ExpireDate is null) 
                        --and t1.StartDate<=finance.addmonth(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),-12)--228
              """

    sql_cmd2 = """select t2.SecuCode as SecuCode 
                        ,t1.TradingDay as TradingDay
                        ,t1.IndexValue as IndexValue
                        ,t1.ValueDailyGrowthRate as ValueDailyGrowthRate
                  from fdm_sor.sor_prd_MF_IndexReturnHis_d t1
                  join
                  fdm_sor.sor_cde_hsfdss_secumain t2
                  on t1.indexcode= t2.innercode
                  where t2.secucode = '000300' and t2.SecuCategory = 4
              """

    sql_cmd3 = """with t1 as 
                (       select innercode
                              ,case when canceldate is null then '2099-01-01' 
                               else canceldate end as canceldate
                              ,maxchargerate
                        from fdm_sor.sor_evt_mf_chargeratenew
                        where chargeratetype in (12000, 12200)  and chargerateunit = 6 
                )   
                ,t2 as 
                (       select innercode
                              ,max(maxchargerate) as ApplyRate
                        from fdm_sor.sor_evt_mf_chargeratenew
                        where chargeratetype in (11210, 11010) and chargerateunit = 6 
                        group by innercode
                ) 
                ,t3 as 
                (       select a.fund_code as fundcode
                              ,b.innercode as innercode
                              ,case when a.apply_rate_discount is null then 0.1
                               else a.apply_rate_discount
                               end as apply_rate_discount
                              ,case when a.redeem_rate_discount is null then 1.0
                               else a.redeem_rate_discount 
                               end as redeem_rate_discount
                              ,row_number() over(partition by a.fund_code order by a.apply_rate_discount) order_nm
                        from fdm_sor.sor_evt_lcpss_pss_product_fund a
                        full outer join fdm_sor.sor_evt_mf_fundarchives b
                        on a.fund_code = b.securitycode
                        where a.fund_code is not null
                )
                select  t3.fundcode as fundcode
                        ,round(t2.ApplyRate*0.01*t3.apply_rate_discount, 5) as Real_ApplyRate
                        ,round(a.maxchargerate*0.01*t3.redeem_rate_discount, 5) as RedeemRate
                from 
                  (select t1.innercode
                         ,t1.maxchargerate
                         ,row_number() over(partition by t1.innercode order by t1.canceldate desc, t1.maxchargerate desc) as order_nm 
                  from t1) a
                join t2
                on a.innercode = t2.innercode
                left join t3
                on a.innercode = t3.innercode
                where a.order_nm = 1 and t3.order_nm = 1
              """

    sql_cmd4 = """select id, upboundary from fbidm.fnd_fund_prft_ls_boundary sort by upboundary"""

    #基金数据
    dataset = operator.execute_sql(sql_cmd1).toPandas()
    boundary = datetime.date(datetime.date.today().year - 1, datetime.date.today().month, datetime.date.today().day)
    dataset.Startdate = dataset.Startdate.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    dataset = dataset[dataset.Startdate <= boundary]
    dataset = dataset[["fundcode", "TradingDay", "nvr", "nvr_growthrate"]]
    #dataset.rename(columns = {"fundcode":"fundcode","TradingDay":"TradingDay","nvr":"nvr","nvr_growthrate":"nvr_growthrate"}, inpalce = True)

    #沪深300市场行情数据
    #dataset_bm = operator.execute_sql(sql_cmd2).toPandas()
    #dataset_bm = dataset_bm.fillna(value = 0).sort('TradingDay').set_index('TradingDay')

    #基金费率
    fee = operator.execute_sql(sql_cmd3).toPandas()
    fee.rename(columns={'fundcode':'fundcode', 'Real_ApplyRate':'apply_fee', 'RedeemRate':'redeem_fee'}, inplace=True)

    #盈亏概率分界表
    rank = operator.execute_sql(sql_cmd4).toPandas()
    rank = rank[['id']]

    return  dataset, fee, rank

if __name__ == '__main__':
    dataset, fee = initial_data()
    print dataset.head(5)
    print fee.head(5)
    print type(dataset.TradingDay[0])

