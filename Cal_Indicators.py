#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Loop according to the input of Fund Code and Fund Indiactors"

from __future__ import unicode_literals
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

__author__ = "WiGi"
__mtime__ = "10 May 2018"

def Extract_Data(dataframe, enddate, years):
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
    if startdate >= dataframe.index[0]:
        return dataframe[startdate:enddate]
    else:
        return None

def Annual_Return(date_line, return_line):
    """
    Formula: (mean(return rates in 1 year)+1)^242 -1
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    :return ret_y: annual average daily return
    """
    df = pd.DataFrame({'trade_date': date_line, 'ret_d': return_line.astype('float') * 0.01}).set_index('trade_date')
    ret_y = (1.0 + df.mean()) ** 242 - 1.0
    return float(ret_y)

def Month_Filter(date_line, capital_line):
    """
    Extract accumulated return of one month except not enough one month's records
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    :return dataset_mret: transform return daily into return monthly
    """
    df = pd.DataFrame({'trade_date': date_line, 'capital': capital_line.astype('float')}).set_index('trade_date')
    dataset_mret = pd.DataFrame(df['capital'].resample('M').last().pct_change()).dropna(axis=0, how='any')
    dataset_mret.columns = ['accum_ret_m']
    dataset_mret['direct_tab'] = np.zeros((len(dataset_mret), 1))
    dataset_mret['direct_tab'].loc[dataset_mret.accum_ret_m > 0] = 1
    dataset_mret['direct_tab'].loc[dataset_mret.accum_ret_m < 0] = -1
    return dataset_mret

def Successive_Rise(date_line, capital_line, enddate):
    """
    Calculate successive rise monthly of a fund within certain years on the point of enddate
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    :param enddate: i.e. nowadays from which to ascend one year records
    """
    dataset_mret = Month_Filter(date_line, capital_line)
    for i in range(len(dataset_mret)):
        month_date = dataset_mret.index[i]
        # try:
        accum_succ_ret_max_dic = {1: None, 3: None, 5: None}
        if month_date.year == enddate.year and month_date.month == enddate.month:
            for years in [1, 3, 5]:
                if (i - 12 * years) >= 0:
                    accum_succ_ret_max = 0.0
                    mret_years = dataset_mret[i - 12 * years:i]
                    j = 0
                    while j < (len(mret_years) - 1):
                        if mret_years.direct_tab[j] != -1:
                            accum_succ_ret = mret_years.accum_ret_m[j]  # initial point
                            for k in range(j + 1, len(mret_years)):
                                if mret_years.direct_tab[k] != -1:
                                    accum_succ_ret = (accum_succ_ret + 1) * (mret_years.accum_ret_m[k] + 1) - 1
                                else:
                                    break
                            j = k + 1
                            if accum_succ_ret > accum_succ_ret_max:
                                accum_succ_ret_max = accum_succ_ret
                        else:
                            j += 1
                else:
                    accum_succ_ret_max = None
                accum_succ_ret_max_dic[years] = accum_succ_ret_max
            break
    return accum_succ_ret_max_dic

def Max_Drawdown(date_line, capital_line):
    """
    Calculate max drawdown of one fund within certain years
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    """
    df = pd.DataFrame({'trade_date': date_line, 'capital': capital_line.astype('float')}).set_index('trade_date')
    df['drawdown_yet'] = df.capital / df.capital.expanding(min_periods=1).max() - 1
    max_drawdown = df['drawdown_yet'].min()
    end_date = df['drawdown_yet'].idxmin()
    start_date = df.loc[:end_date, "capital"].idxmax()
    return max_drawdown, end_date, start_date

def Annual_Vol(date_line, return_line, market_index_ret):
    """
    Calculate annual volatility and annual relative volatility compared with a common market index like HS300
    :param date_line: dates of records in pd.Series or np.narray
    :param return_line: change ratio daily in pd.Series or np.narray
    :param market_line: a certain market index's, benchmark, change ratio daily in pd.Series or np.narray
    """
    df = pd.DataFrame({'trade_date': date_line, 'ret_d': return_line.astype('float') * 0.01,
                       'mret_d': market_index_ret.astype('float') * 0.01}).set_index("trade_date").sort_index()
    df["extra_return"] = df.ret_d - df.mret_d
    anl_voladility = df.ret_d.std() * np.sqrt(242)
    anl_red_voladility = df.extra_return.std() * np.sqrt(242)
    return anl_voladility, anl_red_voladility

def Classify_Allret_2(date_line, capital_line):
    """
    make a tag for a certain fund every valid trading date
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    """
    df = pd.DataFrame({'trade_date': date_line, 'capital': capital_line.astype('float')}).set_index("trade_date").sort_index()
    df = pd.concat([df, pd.DataFrame(columns=list(['end_date', 'end_capital', 'accum_ret']))])
    for hold_dur in [30, 60, 90, 180]:
        df[['end_capital', 'accum_ret']] = None
        for date in df.index:
            if (date + datetime.timedelta(days=hold_dur)) <= df.index.max():
                data_scope = df['capital'][date:(date + datetime.timedelta(days=hold_dur))]
                df.loc[date, 'end_date'] = data_scope.index[-1].strftime('%Y-%m-%d')
                df.loc[date, 'end_capital'] = data_scope.iloc[-1]
        df['accum_ret'] = df['end_capital'] / df['capital'] - 1
        df.loc[df.accum_ret < -0.1, 'accumret_label_' + str(hold_dur)] = 1
        df.loc[(df.accum_ret >= -0.1) & (df.accum_ret < -0.05), 'accumret_label_' + str(hold_dur)] = 2
        df.loc[(df.accum_ret >= -0.05) & (df.accum_ret < 0), 'accumret_label_' + str(hold_dur)] = 3
        df.loc[(df.accum_ret >= 0) & (df.accum_ret < 0.05), 'accumret_label_' + str(hold_dur)] = 4
        df.loc[(df.accum_ret >= 0.05) & (df.accum_ret < 0.1), 'accumret_label_' + str(hold_dur)] = 5
        df.loc[df.accum_ret >= 0.1, 'accumret_label_' + str(hold_dur)] = 6
    return df[['accumret_label_30', 'accumret_label_60', 'accumret_label_90', 'accumret_label_180']]

def Classify_Allret(date_line, capital_line, hold_dur=30):
    """
    make a tag for a certain fund every valid trading date
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    """
    df = pd.DataFrame({'trade_date': date_line, 'capital': capital_line.astype('float')}).set_index("trade_date").sort_index()
    df = pd.concat([df, pd.DataFrame(columns=list(['end_date', 'end_capital', 'accum_ret']))])
    for date in df.index:
        if (date + datetime.timedelta(days=hold_dur)) <= df.index.max():
            data_scope = df['capital'][date:(date + datetime.timedelta(days=hold_dur))]
            df.loc[date, 'end_date'] = data_scope.index[-1].strftime('%Y-%m-%d')
            df.loc[date, 'end_capital'] = data_scope.iloc[-1]
    df['accum_ret'] = df['end_capital'] / df['capital'] - 1
    df.loc[df.accum_ret < -0.1, 'accumret_label'] = 1
    df.loc[(df.accum_ret >= -0.1) & (df.accum_ret < -0.05), 'accumret_label'] = 2
    df.loc[(df.accum_ret >= -0.05) & (df.accum_ret < 0), 'accumret_label'] = 3
    df.loc[(df.accum_ret >= 0) & (df.accum_ret < 0.05), 'accumret_label'] = 4
    df.loc[(df.accum_ret >= 0.05) & (df.accum_ret < 0.1), 'accumret_label'] = 5
    df.loc[df.accum_ret >= 0.1, 'accumret_label'] = 6
    return df.accumret_label

def Classify_Allret_with_Fee(date_line, capital_line, apply_fee, redeem_fee, hold_dur=30):
    """
    make a tag for a certain fund every valid trading date
    :param date_line: dates of records in pd.Series or np.narray
    :param capital_line: net asset value(NAV) in pd.Series or np.narray
    """
    df = pd.DataFrame({'trade_date': date_line, 'capital': capital_line.astype('float')}).set_index("trade_date").sort_index()
    df = pd.concat([df, pd.DataFrame(columns=list(['end_date', 'end_capital', 'accum_ret']))])
    for date in df.index:
        if (date + datetime.timedelta(days=hold_dur)) <= df.index.max():
            data_scope = df['capital'][date:(date + datetime.timedelta(days=hold_dur))]
            df.loc[date, 'end_date'] = data_scope.index[-1].strftime('%Y-%m-%d')
            df.loc[date, 'end_capital'] = data_scope.iloc[-1]
    df['accum_ret'] = (df['end_capital'] * (1.0 - redeem_fee)) / (df['capital'] * (1.0 + apply_fee)) - 1
    df.loc[df.accum_ret < -0.1, 'accumret_label'] = 1
    df.loc[(df.accum_ret >= -0.1) & (df.accum_ret < -0.05), 'accumret_label'] = 2
    df.loc[(df.accum_ret >= -0.05) & (df.accum_ret < 0), 'accumret_label'] = 3
    df.loc[(df.accum_ret >= 0) & (df.accum_ret < 0.05), 'accumret_label'] = 4
    df.loc[(df.accum_ret >= 0.05) & (df.accum_ret < 0.1), 'accumret_label'] = 5
    df.loc[df.accum_ret >= 0.1, 'accumret_label'] = 6
    return df.accumret_label

def Direct_Prob_2(date_line, capital_line, fund_code, window=1):
    """
    make a tag for a certain fund every valid trading date
    :param date_line: dates of records in pd.Series or np.narray
    :param accumret_label_line: a fund's daily tags, which is based on the accumulate profit after a certian holding time, in pd.Series or np.narray
    :param fund_code: the code of a certain fund which is calculated in this function
    :param window: how long to backtrack a certain fund's records
    """
    stat_df = pd.DataFrame()
    df = Classify_Allret(date_line, capital_line)
    for date in df.index:
        df_1y = Extract_Data(df, date, window)  # enough 1 year, valid; others, None
        distrib_ret = pd.DataFrame()
        if df_1y is not None:
            distrib_ret = df_1y.apply(lambda x: x.value_counts() / x.count(),
                                      axis=0)  # Remove None, furthermore may less than 1-6(some certain value's num is 0)
            distrib_ret = distrib_ret.apply(lambda x: x.reindex(index=[1, 2, 3, 4, 5, 6], fill_value=0), axis=0)
        else:
            distrib_ret = pd.DataFrame(
                columns=[u'accumret_label_30', u'accumret_label_60', u'accumret_label_90', u'accumret_label_180'], index=range(1, 7))
        distrib_ret['TradingDate'] = date.strftime('%Y-%m-%d')
        distrib_ret['hold_dur'] = distrib_ret.index
        distrib_ret.set_index('TradingDate', inplace=True)
        stat_df = pd.concat([stat_df, distrib_ret])
    return stat_df

def Direct_Prob(date_line, capital_line, fund_code, apply_fee, redeem_fee, window=1, with_fee=True):
    """
    make a tag for a certain fund every valid trading date
    :param date_line: dates of records in pd.Series or np.narray
    :param accumret_label_line: a fund's daily tags, which is based on the accumulate profit after a certian holding time, in pd.Series or np.narray
    :param fund_code: the code of a certain fund which is calculated in this function
    :param window: how long to backtrack a certain fund's records
    """
    stat_df = []
    if with_fee == False:
        apply_fee = 0.0;
        redeem_fee = 0.0
    for hold_dur in [30, 60, 90, 180]:
        accumret_label = Classify_Allret_with_Fee(date_line, capital_line, apply_fee, redeem_fee, hold_dur)
        df = pd.DataFrame(accumret_label)
        df.columns = ['accumret_label']
        stat_record = []
        for date in df.index:
            df_1y = Extract_Data(df, date, window)  # enough 1 year, valid; others, None
            if df_1y is not None:
                distrib_ret = df_1y.accumret_label.value_counts() / df_1y.accumret_label.count()  # Remove None, furthermore may less than 1-6(some certain value's num is 0)
                distrib_ret = distrib_ret.reindex(index=[1, 2, 3, 4, 5, 6], fill_value=0)
            else:
                distrib_ret = pd.Series(np.repeat(None, 6), index=range(1, 7))
            stat_record = [fund_code, date, hold_dur, distrib_ret[1], distrib_ret[2],
                           distrib_ret[3], distrib_ret[4], distrib_ret[5], distrib_ret[6]]
            stat_df.append(stat_record)
    return pd.DataFrame(stat_df, columns=[u'fund_code', u'TradingDay', u'hold_time',
                                          u'under -10%', u'-10%:-5%', u'-5%:-0%',
                                          u'0%:5%', u'5%:10%', u'over 10%']).set_index('TradingDay')

def Quadratic_Regression_sk(date_line, return_line, market_index_ret, risk_free_profit):
    """
    Calculate Timing Ability based on T-M model, as a kind of Quadratic Regression, based on sklearn function
    :param date_line: dates of records in pd.Series or np.narray
    :param return_line: change ratio daily in pd.Series or np.narray
    :param market_line_ret: a certain market index's return, benchmark, change ratio daily in pd.Series or np.narray
    :param risk_free_profit: change deposit yearly to deposit daily in pd.Series or np.array
    """
    df = pd.DataFrame({'trade_date': date_line, 'ret_d': return_line.astype('float') * 0.01,
                       'mret_d': market_index_ret.astype('float') * 0.01,
                       'rf_ret_d': (1.0 + risk_free_profit.astype(float)) ** (1.0 / 365.0) - 1.0}).set_index(
        "trade_date").sort_index()
    y = np.array(df.ret_d - df.rf_ret_d)
    x1 = np.array(df.mret_d - df.rf_ret_d)
    quadratic_featurizer = PolynomialFeatures(degree=2)
    x2 = quadratic_featurizer.fit_transform(x1.reshape(x1.shape[0], 1))
    TM_model = LinearRegression()
    TM_model.fit(x2, y)
    params = TM_model.coef_
    excess_income = TM_model.intercept_
    beta_no_timing = params[1]
    timing_ability = params[2]
    return excess_income, beta_no_timing, timing_ability

# T-M model： Rp - Rf = alpha + beta*(Rm - Rf) + gamma*(Rm - Rf)**2 + epsilon
def Quadratic_Regression_stat(date_line, return_line, market_index_ret, risk_free_profit):
    """
    Calculate Timing Ability based on T-M model, as a kind of Quadratic Regression, based on statsmodels function
    :param date_line: dates of records in pd.Series or np.narray
    :param return_line: change ratio daily in pd.Series or np.narray
    :param market_line_ret: a certain market index's return, benchmark, change ratio daily in pd.Series or np.narray
    :param risk_free_profit: change deposit yearly to deposit daily in pd.Series or np.array
    """
    df = pd.DataFrame({'trade_date': date_line, 'ret_d': return_line.astype('float') * 0.01,
                       'mret_d': market_index_ret.astype('float') * 0.01,
                       'rf_ret_d': (1.0 + risk_free_profit.astype(float)) ** (1.0 / 365.0) - 1.0}).set_index(
        "trade_date").sort_index()
    y = np.array(df.ret_d - df.rf_ret_d)
    x = np.array(df.mret_d - df.rf_ret_d)
    X = np.column_stack((x, x ** 2))
    X = sm.add_constant(X)  # prepend = False
    model = sm.OLS(y, X)
    results = model.fit()
    excess_income = results.params[0]
    beta_no_timing = results.params[1]
    timing_ability = results.params[2]
    gammastderr = results.bse[2]
    return excess_income, beta_no_timing, timing_ability, gammastderr