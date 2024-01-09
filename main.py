# -*- coding = utf-8 -*-

import os
import warnings
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from connect_database import ConnectDatabase
from bolling_band import BollingBand
from reverse_bolling_band import ReverseBollingBand
from CTABacktester import CTABacktester


# parameters
assets = ['000985.CSI']
positions = [0]
capital = 100000
start_date = '20130101'
end_date = datetime.now().strftime('%Y%m%d')
data_start_date = datetime.strptime(start_date, '%Y%m%d').date() - timedelta(days=700)
data_start_date = data_start_date.strftime('%Y%m%d')
adjust_unit = 1.0
upper_limit = 1.0
lower_limit = -1.0
window = 20
std_width = 2
close_position_at_mean = False
db_wind = {
    'host': '192.168.7.93',
    'port': 3306,
    'username': 'quantchina',
    'password': 'zMxq7VNYJljTFIQ8',
    'database': 'wind'}
bb_description = f"""
              Strategy: Bolling Band \n
              up_curve = mean(close_price, window) + std_width*std(close_price, window); \n
              down_curve = mean(close_price, window) - std_width*std(close_price, window) \n
              When price crosses up_curve, short; When price crosses down_curve, long. \n
              Assets: 000985.CSI \n
              Initial Position: 0.5 \n
              Indicators: close_price, up_curve and down_curve \n
              Parameters: window={window}, std_width={std_width}, close_position_at_mean={close_position_at_mean}  
              """
rbb_description = f"""
              Strategy: Reverse Bolling Band \n
              up_curve = mean(close_price, window) + std_width*std(close_price, window); \n
              down_curve = mean(close_price, window) - std_width*std(close_price, window) \n
              When price crosses up_curve, long; When price crosses down_curve, short. \n
              Assets: 000985.CSI \n
              Initial Position: 0.5 \n
              Indicators: close_price, up_curve and down_curve \n
              Parameters: window={window}, std_width={std_width}, close_position_at_mean={close_position_at_mean}  
              """
benchmark = '000985.CSI'
data_path = os.path.dirname(os.path.abspath(__file__))
save_path_1 = (os.path.dirname(os.path.abspath(__file__))
               + f'/BollingBand({window},{std_width},{close_position_at_mean})_{start_date}_{end_date}.pdf')
save_path_2 = (os.path.dirname(os.path.abspath(__file__))
               + f'/ReverseBollingBand({window},{std_width},{close_position_at_mean})_{start_date}_{end_date}.pdf')

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
for asset, position, data in zip(temp_dict.keys(), positions, temp_dict.values()):
    bb = BollingBand(asset, data, position, capital, start_date, end_date, adjust_unit,
                             upper_limit, lower_limit, window, std_width, data_path, close_position_at_mean)
    rbb = ReverseBollingBand(asset, data, position, capital, start_date, end_date, adjust_unit,
                             upper_limit, lower_limit, window, std_width, data_path, close_position_at_mean)
    bb.run()
    rbb.run()

bb_backtester = CTABacktester(bb_description, assets, positions, capital, start_date, end_date,
                              benchmark, data_path, save_path_1)
rbb_backtester = CTABacktester(rbb_description, assets, positions, capital, start_date, end_date,
                               benchmark, data_path, save_path_2)
bb_backtester.run_backtest()
rbb_backtester.run_backtest()
