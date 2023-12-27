# -*- coding = utf-8 -*-
# @Time: 2023/12/19 16:30
# @Author: Jiahao Xu
# @File: bolling_band.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Connect_Database.connect_database_new import ConnectDatabase
from Project2_Strategy.Backtester import CTABacktester


class BollingBand:
    """ """
    def __init__(self, asset, weight, data, start_date, end_date, adjust_unit, upper_limit, lower_limit, window,
                 std_range, save_path):
        """
        : param asset:
        : param weight:
        : param start_date:
        : param end_date:
        : param asset_data:
        : param adjust_unit:
        : param upper_limit:
        : param lower_limit:
        : param window:
        : param range:
        """
        self.asset = asset
        self.weight = weight
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.adjust_unit = adjust_unit
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.window = window
        self.std_range = std_range
        self.save_path = save_path

    def cal_bar(self):
        """ """
        # self.data.reset_index(drop=True, inplace=True)
        rolling_mean = self.data['close'].rolling(self.window).mean()
        rolling_std = self.data['close'].rolling(self.window).std(ddof=1)
        upper_limit = rolling_mean + self.std_range * rolling_std
        lower_limit = rolling_mean - self.std_range * rolling_std

        return rolling_mean, upper_limit, lower_limit

    def trade_signals(self):
        """ """
        self.data.reset_index(drop=True, inplace=True)
        self.data['ma'], self.data['up'], self.data['down'] = self.cal_bar()
        self.data.loc[:, 'signal'] = np.where((self.data['close'] > self.data['up'])
                                       & (self.data['close'].shift(1) < self.data['up'].shift(1)), -1,
                                       np.where((self.data['close'] < self.data['down'])
                                       & (self.data['close'].shift(1) > self.data['down'].shift(1)), 1, 0))
        self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]
        self.data.reset_index(inplace=True, drop=True)
        self.data.loc[0, 'weight'] = self.weight
        for i in range(len(self.data) - 1):
            signal = self.data.loc[i, 'signal']
            position = self.data.loc[i, 'weight']

            if signal == 1:
                self.data.loc[i + 1, 'weight'] = np.round(min(position + self.adjust_unit, self.upper_limit), 1)
            elif signal == -1:
                self.data.loc[i + 1, 'weight'] = np.round(max(position - self.adjust_unit, self.lower_limit), 1)
            else:
                self.data.loc[i + 1, 'weight'] = np.round(position, 1)

        return self.data.sort_values(['symbol', 'date'])  # [['symbol', 'date', 'return', 'weight']]

    def run(self):
        """ """
        df = self.trade_signals()
        df.to_parquet(self.save_path + f'/{self.asset}_{self.start_date}_{self.end_date}.parquet')


if __name__ == '__main__':
    # parameters
    assets = ['000985.CSI']
    weights = [0.5]
    start_date = '20130101'
    end_date = datetime.now().strftime('%Y%m%d')
    data_start_date = datetime.strptime(start_date, '%Y%m%d').date() - timedelta(days=700)
    data_start_date = data_start_date.strftime('%Y%m%d')
    adjust_unit = 1.0
    upper_limit = 1.0
    lower_limit = 0.0
    window = 20
    std_range = 1
    db_wind = {
        'host': '192.168.7.93',
        'port': 3306,
        'username': 'quantchina',
        'password': 'zMxq7VNYJljTFIQ8',
        'database': 'wind'}
    description = """
                  Strategy: Bolling Band \n
                  Assets: 000985.CSI \n
                  Initial Position: 0.5 \n
                  Indicators: close_price, up_curve and down_curve \n
                  when price crosses up_curve, short; when price crosses down_curve, long.
                  """
    benchmark = '000985.CSI'
    data_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.dirname(os.path.abspath(__file__)) + f'/BolllingBand_{start_date}_{end_date}.pdf'

    # get_data
    temp_dict = dict()
    for asset in assets:
        table = 'AINDEXEODPRICES'
        columns = 'S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_AMOUNT'
        condition1 = f"S_INFO_WINDCODE = '{asset}'"
        condition2 = f"TRADE_DT BETWEEN '{data_start_date}'AND '{end_date}'"
        sql = f''' SELECT %s FROM %s WHERE %s AND %s ''' % (columns, table, condition1, condition2)
        cd = ConnectDatabase(db_wind, sql)
        data = cd.get_data()
        data = data.rename(columns={'S_INFO_WINDCODE': 'symbol', 'TRADE_DT': 'date', 'S_DQ_PRECLOSE': 'pre_close',
                                  'S_DQ_OPEN': 'open', 'S_DQ_HIGH': 'high', 'S_DQ_LOW': 'low', 'S_DQ_CLOSE': 'close',
                                  'S_DQ_AMOUNT': 'amount'})
        data[data.columns[2:]] = (data[data.columns[2:]].apply(pd.to_numeric))
        data = data.sort_values(['symbol', 'date']).copy()
        data['return'] = data['close']/data['pre_close'] - 1
        temp_dict[asset] = data

    # run bolling band strategy
    for asset, weight, data in zip(temp_dict.keys(), weights, temp_dict.values()):
        bb = BollingBand(asset, weight, data, start_date, end_date, adjust_unit, upper_limit, lower_limit,
                         window, std_range, data_path)
        bb.run()

    backtester = CTABacktester(description, assets, weights, start_date, end_date, benchmark, data_path, save_path)
    backtester.run_backtest()
