#!usr/bin/env python
#-*- coding=utf-8 -*-

from __future__ import unicode_literals
from uuid import uuid1
from spark_operator import operator, spark
from pyspark.sql.types import StructField, StructType, StringType, FloatType, IntegerType, BooleanType, DateType
from pyspark.sql.functions import when, col
import warnings
warnings.filterwarnings('ignore')

__author__ = 'WiGi'
__mtime__ = 'June 6, 2018'

def Trans_Struct(Table_Name):
    if Table_Name == 'fbidm.fnd_fund_hd_fxd_prid_return_standby':
        return StructType([
            StructField('Id', StringType(), False),#whehther can be Null
            StructField('FundCode', StringType(), False),
            StructField('PeriodReturn', FloatType(), True),
            StructField('HoldingPeriod', FloatType(), False),#IntegerType()
            StructField('StartDate', StringType(), True), #maybe nan, so from datetype() to stringtype()
            StructField('EndDate', DateType(), False), #Datetype() can't accept NaN from pd.DataFrame.date()
            StructField('UpdateTime', StringType(), False)
        ])
    elif Table_Name == 'fbidm.fnd_fund_future_potential_standby':
        return StructType([
            StructField('Id', StringType(), False),#whehther can be Null
            StructField('FundCode', StringType(), False),
            StructField('StartDate', StringType(), True),#maybe nan, so from datetype() to stringtype()
            StructField('EndDate', DateType(), False),#Datetype() can't accept NaN from pd.DataFrame.date()
            StructField('Fk', StringType(), False),
            StructField('SampleNum', StringType(), False),
            StructField('TotalNum', StringType(), False),
            StructField('Probability', FloatType(), True),
            StructField('HoldingPeriod', IntegerType(), False),
            StructField('Period', IntegerType(), False),
            StructField('IfFee', BooleanType(), False),
            StructField('UpdateTime', StringType(), False)
        ])


def Add_UUID(df):
    """
    :param df:dataframe which will be writed into Hive
    :return:dataframe added uuid column
    """
    ID = []
    for i in range(len(df)):
        ID.append(uuid1().hex)
    df['Id'] = ID
    return df

def Save_Table(pd_df, Table_Name):
    """
    :param pd_df: data which will be save into hive
    :param Table_Name: table's name as which data will be saved
    """
    sp_df = spark.createDataFrame(pd_df, schema=Trans_Struct(Table_Name))
    for i in sp_df.columns:
        sp_df = sp_df.withColumn(i, when(col(i) == 'NaN', None).otherwise(col(i)))
    sp_df.registerTempTable('temp_sp_df')
    #operator.execute_sql("select * from temp_sp_df")
    sp_df.show()
    operator.drop_table(Table_Name)
    spark.sql("create table if not exists %s as select * from temp_sp_df"%(Table_Name))

def Insert_Table(pd_df, Table_Name):
    sp_df = spark.createDataFrame(pd_df, schema=Trans_Struct(Table_Name))
    for i in sp_df.columns:
        sp_df = sp_df.withColumn(i, when(col(i) == 'NaN', None).otherwise(col(i)))
    #sp_df.registerTempTable('temp_sp_df')
    #sp_df.write.insertInto(Table_Name, False)
    operator.save(table_name=Table_Name, data_frame=sp_df, overwrite=False)

if __name__ == '__main__':
    pass
