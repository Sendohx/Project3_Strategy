
import pandas as pd
import numpy as np
import itertools
import warnings
from datetime import datetime, timedelta
from bolling_band import BollingBand
from reverse_bolling_band import ReverseBollingBand
warnings.filterwarnings('ignore')

root = '/nas92/xujiahao'
strategy1 = 'BollingBand'
strategy2 = 'ReverseBollingBand'
assets = ['000985.CSI', '000300.SH', '000852.SH', '000905.SH']
positions = [0, 0, 0, 0]
capital = 1000000
start_date = '20130101'
end_date = datetime.today().strftime('%Y%m%d')
data_start_date = datetime.strptime(start_date, '%Y%m%d').date() - timedelta(days=700)
data_start_date = data_start_date.strftime('%Y%m%d')
adjust_unit = 1.0
upper_limit = 1.0
lower_limit = -1.0
param_grid={
    'window': list(range(10,20,1)),
    'std_width': list(np.round(np.arange(0.5,2.1,0.25), 1)),
    'close_position_at_mean': [True, False]}
data_path = root + f'/data/bollingband'

# get_data
temp_dict = dict()
for asset in assets:
    data = pd.read_parquet(root + f'/data/raw/ind_{asset}_{start_date}_{end_date}.parquet')
    data = data[data['date']<='20231231']
    temp_dict[asset] = data

# run bolling band strategy
for asset, position, data in zip(temp_dict.keys(), positions, temp_dict.values()):
    bb_sum = pd.DataFrame(columns=['window', 'std_width', 'close_at_mean',
                                '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                                '2020', '2021', '2022', '2023','all_time',
                                '13','14','15','16','17','18','19','20','21','22','23','alltime','num'])
    rbb_sum = pd.DataFrame(columns=['window', 'std_width', 'close_at_mean',
                                   '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                                   '2020', '2021', '2022', '2023', 'all_time',
                                   '13','14','15','16','17','18','19','20','21','22','23','alltime','num'])
    for window, std_width, close in itertools.product(*param_grid.values()):
        bb = BollingBand(asset, data, position, capital, start_date, end_date, adjust_unit,
                                 upper_limit, lower_limit, window, std_width, data_path, close)
        rbb = ReverseBollingBand(asset, data, position, capital, start_date, end_date, adjust_unit,
                                 upper_limit, lower_limit, window, std_width, data_path, close)
        df_bb = bb.cal_return()
        df_rbb = rbb.cal_return()
        df_bb['year'] = df_bb['date'].str[:4]
        df_rbb['year'] = df_bb['date'].str[:4]
        #df_bb['real_ret'] = df_bb['real_ret'].round(9).replace([0.000000000, -0.000000000], int(0))
        #df_rbb['real_ret'] = df_rbb['real_ret'].round(9).replace([0.000000000,-0.000000000],int(0))
        bb_win_list = df_bb.groupby('year')['real_ret'].apply(lambda x: np.round((x > 0).sum()/(x != 0).sum()*100,1)).tolist()
        rbb_win_list = df_rbb.groupby('year')['real_ret'].apply(lambda x: np.round((x > 0).sum()/(x != 0).sum()*100,1)).tolist()
        bb_ret_list = df_bb.groupby('year')['real_ret'].sum().tolist()
        rbb_ret_list = df_rbb.groupby('year')['real_ret'].sum().tolist()
        bb_trade_num = (df_bb['position'].diff() != 0).sum()
        rbb_trade_num = (df_rbb['position'].diff() != 0).sum()
        try:
            bb_sum = bb_sum._append(
                pd.DataFrame(data =[[window, std_width, close] + bb_win_list
                                    + [df_bb['real_ret'].apply(lambda x: np.round((x > 0).sum()/(x != 0).sum(),1))]
                                    + bb_ret_list + [df_bb['real_ret'].mean()*252, bb_trade_num]],
                             columns=['window', 'std_width', 'close_at_mean',
                                      '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                                      '2020', '2021', '2022', '2023', 'all_time',
                                      '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'alltime','num']),
                             ignore_index=True)
            rbb_sum = rbb_sum._append(
                pd.DataFrame(data=[[window, std_width, close] + rbb_win_list
                                   + [df_rbb['real_ret'].apply(lambda x: np.round((x > 0).sum() / (x != 0).sum(), 1))]
                                   + rbb_ret_list + [df_rbb['real_ret'].mean()*252, rbb_trade_num]],
                             columns=['window', 'std_width', 'close_at_mean',
                                      '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                                      '2020', '2021', '2022', '2023', 'all_time',
                                      '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'alltime','num']),
                             ignore_index=True)
            print(f'{asset}_{window}_{std_width}_{close} finish')
        except Exception as e:
            print(f'{e} caused by {asset},{window},{std_width},{close}')

    columns = pd.MultiIndex.from_tuples([('param','window'), ('param', 'std_width'), ('param', 'close_at_mean'),
                                         ('win_rate(%)','2013'), ('win_rate(%)','2014'), ('win_rate(%)','2015'),
                                         ('win_rate(%)','2016'), ('win_rate(%)','2017'), ('win_rate(%)','2018'),
                                         ('win_rate(%)','2019'), ('win_rate(%)','2020'), ('win_rate(%)','2021'),
                                         ('win_rate(%)','2022'), ('win_rate(%)','2023'),('win_rate(%)','all_time'),
                                         ('annual_ret(%)', '2013'),('annual_ret(%)', '2014'),('annual_ret(%)', '2015'),
                                         ('annual_ret(%)', '2016'),('annual_ret(%)', '2017'),('annual_ret(%)', '2018'),
                                         ('annual_ret(%)', '2019'),('annual_ret(%)', '2020'),('annual_ret(%)', '2021'),
                                         ('annual_ret(%)', '2022'),('annual_ret(%)', '2023'),('annual_ret(%)','all_time'),
                                         ('trade','number')])
    bb_sum.columns = columns
    rbb_sum.columns = columns
    bb_sum.to_parquet(data_path + f'/{asset}_bb_summary.parquet')
    rbb_sum.to_parquet(data_path + f'/{asset}_rbb_summary.parquet')
