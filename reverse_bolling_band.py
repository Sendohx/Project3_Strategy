# -*- coding = utf-8 -*-
# @Time: 2023/12/19 16:30
# @Author: Jiahao Xu
# @File: reverse_bolling_band.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Connect_Database.connect_database_new import ConnectDatabase
from Project3_Strategy.CTABacktester import CTABacktester


class ReverseBollingBand:
    """ """
    def __init__(self, asset, data, init_position, capital, start_date, end_date, adjust_unit, upper_limit, lower_limit
                 , window, std_width, save_path, close_position_at_mean=False):
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
        self.data = data
        self.init_position = init_position
        self.capital = capital
        self.start_date = start_date
        self.end_date = end_date
        self.adjust_unit = adjust_unit
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.window = window
        self.std_width = std_width
        self.save_path = save_path
        self.close_position_at_mean = close_position_at_mean

    def cal_bar(self):
        """ """
        # self.data.reset_index(drop=True, inplace=True)
        rolling_mean = self.data['close'].rolling(self.window).mean()
        rolling_std = self.data['close'].rolling(self.window).std(ddof=1)
        upper_limit = rolling_mean + self.std_width * rolling_std
        lower_limit = rolling_mean - self.std_width * rolling_std

        return rolling_mean, upper_limit, lower_limit

    def trade_signals(self):
        """ trade_signal, position"""
        self.data.reset_index(drop=True, inplace=True)
        self.data['ma'], self.data['up'], self.data['down'] = self.cal_bar()
        if self.close_position_at_mean is not True:
            self.data.loc[:, 'trade_signal'] = np.where((self.data['close'] > self.data['up'])
                                           & (self.data['close'].shift(1) < self.data['up'].shift(1)), 'long',
                                           np.where((self.data['close'] < self.data['down'])
                                           & (self.data['close'].shift(1) > self.data['down'].shift(1)), 'short', 'keep'))
            # self.data[self.data['return'] <= self.max_single_loss]['trade_signal'] = 'close'
            self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]
            self.data.reset_index(inplace=True, drop=True)
            self.data.loc[0, 'position'] = self.init_position
            for i in range(1, len(self.data)):
                ret = self.data.loc[i-1, 'return']
                trade_signal = self.data.loc[i-1, 'trade_signal']
                position = self.data.loc[i-1, 'position']
                if trade_signal == 'long':
                    self.data.loc[i, 'position'] = np.round(min(position + self.adjust_unit, self.upper_limit), 1)
                elif trade_signal == 'short':
                    self.data.loc[i, 'position'] = np.round(max(position - self.adjust_unit, self.lower_limit), 1)
                elif trade_signal == 'keep':
                    self.data.loc[i, 'position'] = np.round(position, 1)
                elif trade_signal == 'close':
                    self.data.loc[i, 'position'] = 0
        else:
            self.data.loc[:, 'signal'] = np.where((self.data['close'] > self.data['up'])
                                         & (self.data['close'].shift(1) < self.data['up'].shift(1)), 'long',
                                         np.where((self.data['close'] < self.data['down'])
                                         & (self.data['close'].shift(1) > self.data['down'].shift(1)), 'short',
                                         np.where((self.data['close'] < self.data['ma'])
                                         & (self.data['close'].shift(1) > self.data['ma'].shift(1)), 'close',
                                         np.where((self.data['close'] > self.data['ma'])
                                         & (self.data['close'].shift(1) < self.data['ma'].shift(1)), 'close', 'keep'))))
            # self.data[self.data['return'] <= self.max_single_loss]['trade_signal'] = 'close'
            self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]
            self.data.reset_index(inplace=True, drop=True)
            self.data.loc[0, 'position'] = self.init_position
            for i in range(1, len(self.data)):
                signal = self.data.loc[i-1, 'signal']
                position = self.data.loc[i-1, 'position']
                if signal == 'long':
                    self.data.loc[i, 'position'] = np.round(min(position + self.adjust_unit, self.upper_limit), 1)
                elif signal == 'short':
                    self.data.loc[i, 'position'] = np.round(max(position - self.adjust_unit, self.lower_limit), 1)
                elif signal == 'keep':
                    self.data.loc[i, 'position'] = np.round(position, 1)
                elif signal == 'close':
                    self.data.loc[i, 'position'] = 0

        return self.data.sort_values(['symbol', 'date'])  # [['symbol', 'date', 'return', 'weight']]

    def cal_return(self):
        self.trade_signals()
        self.data.loc[0, 'net'] = self.capital
        self.data.loc[0, 'abs_ret'] = self.data.loc[0, 'net'] * self.data.loc[0, 'position'] * self.data.loc[0, 'return']
        self.data.loc[0, 'real_ret'] = self.data.loc[0, 'abs_ret'] / self.capital
        for i in range(1, len(self.data)):
            self.data.loc[i, 'net'] = self.data.loc[i-1, 'net'] + self.data.loc[i-1, 'abs_ret']  # not available for portfolio
            self.data.loc[i, 'abs_ret'] = (self.data.loc[i, 'net'] * self.data.loc[i, 'position']
                                           * self.data.loc[i, 'return'])
            self.data.loc[i, 'real_ret'] = self.data.loc[i, 'abs_ret'] / self.capital
        return self.data

    def run(self):
        """ """
        df = self.cal_return()
        #df = pd.merge(df1, df2, on='date', how='outer')
        df.to_parquet(self.save_path + f'/{self.asset}_{self.start_date}_{self.end_date}.parquet')
