#!usr/bin/evt python
#-*- coding:utf-8 -*-

"Calculate the probability of up and down"

from __future__ import unicode_literals
from __future__ import division
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

__author__ = "WiGi"
__mtime__ = "June 5, 2018"

class cal_dirct_prob(object):

    def __init__(self, index, nvr, fundcode, apply_fee, redeem_fee):
        self.date_line = index
        self.capital_line = nvr
        self.apply_fee = apply_fee
        self.redeem_fee = redeem_fee
        self.__fund_code = fundcode
        self.__stat_df = []
        self.__holding_period = np.array([30, 60, 90, 180]) #holding days
        self.df_multiple = pd.DataFrame()
        self.__date = index[-1]

    def Extract_Data(self, df, enddate, years):
        """
        rolling to extact a chunk of data between 'enddata-year' and 'enddate'
        :param datafram: a certain fund's history data in pandas.DataFrame
        :parma enddate: i.e. nowadays from which to ascend one year records
        :return dataframe: data in dataframe extraced from the initial data of a fund
        """
        try:
            startdate = datetime.datetime(enddate.year - years, enddate.month, enddate.day)
        except ValueError:
            # Note: There is different days in February in common year and leap year, respectively 28 and 29
            startdate = datetime.datetime(enddate.year - years, enddate.month, enddate.day - 1)

        if startdate >= df.index[0]:
            return df[startdate:enddate]
        else:
            return None

    def Calculate_Accumret(self, with_fee = False):#default: False
        """
        :return list_multiple: one record which will be inserted into table fbidm.fnd_fund_hd_fxd_prid_return_standby
        """
        df_initial = pd.DataFrame({'trade_date': self.date_line, 'capital': self.capital_line.astype('float')}).set_index("trade_date").sort_index()
        list_multiple = []

        if with_fee == False:
            self.apply_fee = 0.0; self.redeem_fee = 0.0
        for hold_dur in [30, 60, 90, 180]:
            df = df_initial.copy()
            if (self.__date + datetime.timedelta(days=-hold_dur)) >= df.index.min():
                data_scope = df['capital'][(self.__date + datetime.timedelta(days=-hold_dur)):self.__date]
                start_date = data_scope.index[0].strftime('%Y-%m-%d')

            accum_ret = ((data_scope.iloc[-1] * (1.0 - self.redeem_fee)) / (data_scope.iloc[0] * (1.0 + self.apply_fee)) - 1.0).astype('float')
            holding_period = np.argwhere(self.__holding_period == hold_dur)[0,0]+1
            list_multiple.append([self.__fund_code, holding_period, start_date, accum_ret])
        return list_multiple

    def Classify_Allret_with_Fee(self, with_fee = True):
        """
        make a tag for a certain fund every valid trading date
        :param date_line: dates of records in pd.Series or np.narray
        :param capital_line: net asset value(NAV) in pd.Series or np.narray
        """
        df_initial = pd.DataFrame({'trade_date': self.date_line, 'capital': self.capital_line.astype('float')}).set_index("trade_date").sort_index()
        df_initial = pd.concat([df_initial, pd.DataFrame(columns=list(['start_date', 'start_capital', 'accum_ret']))])

        if with_fee == False:
            self.apply_fee = 0.0; self.redeem_fee = 0.0

        for hold_dur in [30, 60, 90, 180]:
            df = df_initial.copy()
            for date in df.index:
                if (date + datetime.timedelta(days = -hold_dur)) >= df.index.min():
                    data_scope = df['capital'][(date + datetime.timedelta(days = -hold_dur)):date]
                    df.loc[date, 'start_date'] = data_scope.index[0].strftime('%Y-%m-%d')
                    df.loc[date, 'start_capital'] = data_scope.iloc[0]

            df['accum_ret'] = ((df['capital'] * (1.0 - self.redeem_fee)) / (df['start_capital'] * (1.0 + self.apply_fee)) - 1.0).astype('float')
            df.loc[df.accum_ret < -0.1, 'accumret_label'] = 1
            df.loc[(df.accum_ret >= -0.1) & (df.accum_ret < -0.05), 'accumret_label'] = 2
            df.loc[(df.accum_ret >= -0.05) & (df.accum_ret < 0), 'accumret_label'] = 3
            df.loc[(df.accum_ret >= 0) & (df.accum_ret < 0.05), 'accumret_label'] = 4
            df.loc[(df.accum_ret >= 0.05) & (df.accum_ret < 0.1), 'accumret_label'] = 5
            df.loc[df.accum_ret >= 0.1, 'accumret_label'] = 6
            df['holding_period'] = np.argwhere(self.__holding_period == hold_dur)[0,0]+1
            df['fund_code'] = self.__fund_code
            self.df_multiple = pd.concat([self.df_multiple, df[['fund_code', 'start_date', 'accum_ret', 'accumret_label', 'holding_period']]])
        return self.df_multiple

    def Direct_Prob(self, window = 1, with_fee = True):
        """
        make a tag for a certain fund every valid trading date
        :param date_line: dates of records in pd.Series or np.narray
        :param accumret_label_line: a fund's daily tags, which is based on the accumulate profit after a certian holding time, in pd.Series or np.narray
        :param fund_code: the code of a certain fund which is calculated in this function
        :param window: how long to backtrack a certain fund's records
        """
        #df_multiple = self.Classify_Allret_with_Fee(with_fee)
        self.__stat_df = []
        df_slices = self.df_multiple.groupby("holding_period")
        for hold_dur, data_group in df_slices:
            df_tmp = data_group.loc[:,['accumret_label']]
            for date in df_tmp.index:
                df_ys = self.Extract_Data(df_tmp, date, window)
                if df_ys is not None:
                    sample_num = df_ys.accumret_label.value_counts().reindex(index=[1, 2, 3, 4, 5, 6], fill_value=0)
                    total_num = df_ys.accumret_label.count()
                    distrib_ret = (sample_num / total_num).astype('float')

                    for i in range(1,7):
                        __stat_record = [self.__fund_code, date.date(), hold_dur,
                                         df_ys.index[0].strftime('%Y-%m-%d'), str(total_num),#data_group.start_date[date], (int(np.argwhere(self.__period == self.__window))+1)
                                        window, with_fee]+\
                                        [str(sample_num[i]), distrib_ret[i], i]
                        self.__stat_df.append(__stat_record)
        return self.__stat_df

    def Direct_Prob_Add(self, window = 1, with_fee = True):
        """
        make a tag for a certain fund every valid trading date
        :param date_line: dates of records in pd.Series or np.narray
        :param accumret_label_line: a fund's daily tags, which is based on the accumulate profit after a certian holding time, in pd.Series or np.narray
        :param fund_code: the code of a certain fund which is calculated in this function
        :param window: how long to backtrack a certain fund's records
        """
        self.__stat_df = []
        df_slices = self.df_multiple.groupby("holding_period")
        for hold_dur, data_group in df_slices:
            df_tmp = data_group.loc[:,['accumret_label']]
            df_ys = self.Extract_Data(df_tmp, self.__date, window)
            if df_ys is not None:
                sample_num = df_ys.accumret_label.value_counts().reindex(index=[1, 2, 3, 4, 5, 6], fill_value=0)
                total_num = df_ys.accumret_label.count()
                distrib_ret = (sample_num / total_num).astype('float')

                for i in range(1, 7):
                    __stat_record = [self.__fund_code, self.__date.date(), hold_dur,
                                     df_ys.index[0].strftime('%Y-%m-%d'), str(total_num),#data_group.start_date[self.__date]
                                     window, with_fee] + \
                                    [str(sample_num[i]), distrib_ret[i], i]
                    self.__stat_df.append(__stat_record)
        return self.__stat_df

if __name__ == '__main__':
    Cal_Dirct_Prob = cal_dirct_prob()
    stat_df = Cal_Dirct_Prob.Direct_Prob()
    print stat_df[:5]
