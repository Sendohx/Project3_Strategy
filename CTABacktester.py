# -*- coding = utf-8 -*-
# @Time: 2023/12/19 12:52
# @Author: Jiahao Xu
# @File: CTABacktester.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Grid
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CTABacktester:
    """ """
    def __init__(self, description, assets, positions, capital, start_date, end_date, benchmark, data_path, save_path,
                 min_acceptable_return=0.0):
        self.description = description
        self.assets = assets
        self.positions = positions
        self.capital = capital
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.data_path = data_path
        self.save_path = save_path
        self.min_acceptable_return = min_acceptable_return
        self.benchmark_returns = None
        self.strategy_returns = None
        self.excess_returns = None

    def get_benchmark_data(self):
        """ """
        benchmark_data = pd.read_parquet(
            self.data_path + f'/{self.benchmark}_{self.start_date}_{self.end_date}.parquet')
        # benchmark_data['date'] = pd.to_datetime(benchmark_data['date'], format='%Y%m%d')
        #benchmark_data = benchmark_data.loc[
        #    (benchmark_data['date'] >= self.start_date) & (benchmark_data['date'] <= self.end_date)]
        benchmark_data.set_index('date', inplace=True)
        return benchmark_data

    def get_asset_data(self, asset):
        """ """
        asset_data = pd.read_parquet(self.data_path + f'/{asset}_{self.start_date}_{self.end_date}.parquet')
        # asset_data['date'] = pd.to_datetime(asset_data['date'], format='%Y%m%d')
        # asset_data = asset_data.loc[(asset_data['date'] >= self.start_date) & (asset_data['date'] <= self.end_date)]
        asset_data.set_index('date', inplace=True)
        return asset_data

    def get_portfolio_returns(self):
        """ """
        portfolio_returns = None
        portfolio_positions = None
        positions_desc = None

        for asset in self.assets:
            asset_data = self.get_asset_data(asset)
            # return_data = asset_data['weighted_return']
            # return_data = return_data.rename(columns={'weighted_return':'return'})
            mean_weight = asset_data['position'].mean()
            position_changes = asset_data['position'].diff() != 0
            adjustment_frequency = position_changes.sum() / len(asset_data)
            temp_series = pd.Series([mean_weight, adjustment_frequency], name=asset)
            positions_desc = pd.concat([positions_desc, temp_series], axis=1)

            if portfolio_returns is None:
                portfolio_returns = asset_data['real_ret']
            else:
                portfolio_returns += asset_data['real_ret']

            if portfolio_positions is None:
                portfolio_positions = asset_data['position']
            else:
                portfolio_positions += asset_data['position']

        # portfolio_returns.name = 'return'
        # positions_desc = positions_desc.iloc[:, 1:]
        positions_desc.index = ['average_position', 'adjust_frequency']
        positions_desc = positions_desc.round(4)
        positions_desc = positions_desc.T
        positions_desc.index.name = 'asset'
        positions_desc.reset_index(inplace=True)
        return portfolio_returns, positions_desc, portfolio_positions

    def execute_trades(self):
        """ """
        self.strategy_returns = self.get_portfolio_returns()[0]
        self.benchmark_returns = self.get_benchmark_data()['return']
        self.positions = self.get_portfolio_returns()[2]
        # self.excess_returns = self.strategy_returns - self.benchmark_returns
        # self.strategy_returns.iloc[0] = 0
        self.benchmark_returns.iloc[0] = 0
        # self.excess_returns.iloc[0] = 0

    def plot_performance(self):
        """ """
        fig1 = plt.figure(figsize=(25, 10))
        plt.rcParams['font.size'] = 16
        plt.rcParams["figure.autolayout"] = True
        plt.plot(self.strategy_returns.cumsum() + 1, label='Strategy Returns')  # net return
        plt.plot((self.benchmark_returns + 1).cumprod(), label='Benchmark Returns')
        # plt.plot((self.excess_returns + 1).cumprod(), label='Excess Returns')
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))
        plt.title('Return Trend')
        # plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Cumulative \n Returns', rotation=0, labelpad=40)
        # plt.yticks(rotation=45)

        return fig1

    def plot_positions_bar(self):
        fig2 = plt.figure(figsize=(25, 7))
        plt.rcParams["figure.autolayout"] = True
        plt.plot(self.get_portfolio_returns()[2], label='position')
        # plt.legend(loc='best')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))
        plt.xticks(rotation=45)
        plt.title('Everyday_Position')

        return fig2

    def plot_cumulative_returns(self):
        y_data_1 = self.strategy_returns.cumsum() + 1
        y_data_2 = (1 + self.benchmark_returns).cumprod()
        y_bar_data = self.positions.tolist()

        y_line_min = round((min(y_data_1.min(), y_data_2.min()) - 0.002), 3)
        y_line_max = round((max(y_data_1.max(), y_data_2.max()) + 0.002), 3)
        y_bar_min = -1
        y_bar_max = 1

        datazoom_opts = [
            opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1]),
            opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1])
        ]

        line = Line()
        line.add_xaxis(xaxis_data=self.strategy_returns.index.tolist())
        line.add_yaxis(
            series_name="strategy return",
            y_axis= 1 + self.strategy_returns.cumsum(),
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=2),
            label_opts=opts.LabelOpts(is_show=False),
        )
        line.add_yaxis(
            series_name="benchmark return",
            y_axis= (self.benchmark_returns + 1).cumprod(),
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=2),
            label_opts=opts.LabelOpts(is_show=False),
        )

        bar = Bar()
        bar.add_xaxis(xaxis_data=self.positions.index.tolist())
        bar.add_yaxis(
            series_name="position",
            y_axis=y_bar_data,
            label_opts=opts.LabelOpts(is_show=False)
        )

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{self.start_date}_{self.end_date}_trend"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_left="right"),
            datazoom_opts=datazoom_opts,
            yaxis_opts=opts.AxisOpts(type_="value", min_=y_line_min, max_=y_line_max),
        )

        bar.set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            legend_opts=opts.LegendOpts(pos_left="middle"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(type_="value", min_=y_bar_min, max_=y_bar_max),
            datazoom_opts=datazoom_opts
        )

        # create grid item
        grid_chart = Grid()

        grid_chart.add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="5%",
                pos_right="5%",
                pos_top="75%",
                height="20%"
            ),
        )

        grid_chart.add(
            line,
            grid_opts=opts.GridOpts(
                pos_left="5%",
                pos_right="5%",
                pos_top="10%",
                height="60%"
            ),
        )
        grid_chart.render(os.path.splitext(self.save_path)[0] + f".html")

    def simple_annual_return(self, returns):
        """ """
        return returns.mean() * 242

    def compound_annual_return(self, returns):
        """ """
        return ((returns + 1).prod()) ** (242 / len(returns)) - 1

    def annual_volatility(self, returns):
        """ """
        return returns.std() * np.sqrt(242)

    def simple_max_drawdown(self, returns):
        """ """
        cum_returns = returns.cumsum() + 1
        peak = cum_returns.cummax()
        drawdown = (peak - cum_returns)/peak * 100
        max_drawdown = drawdown.max()
        return max_drawdown

    def compound_max_drawdown(self, returns):
        """ """
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (peak - cum_returns)/peak * 100
        max_drawdown = drawdown.max()
        return max_drawdown
    def sharpe_ratio(self, returns, simple=True):
        """ """
        if simple is True:
            return self.simple_annual_return(returns) / self.annual_volatility(returns)
        else:
            return self.compound_annual_return(returns) / self.annual_volatility(returns)

    def sortino_ratio(self, returns, simple=True):
        """ """
        downside_volatility = returns[returns < self.min_acceptable_return].std() * np.sqrt(242)
        if simple is True:
            return self.simple_annual_return(returns) / downside_volatility
        else:
            return self.compound_annual_return(returns) / downside_volatility

    def get_dataframe(self):
        """ """
        metrics = dict()

        metrics['Annual_Returns'] = [
            self.simple_annual_return(self.strategy_returns),
            self.compound_annual_return(self.benchmark_returns)]
            #self.calculate_annual_return(self.excess_returns)]

        metrics['Annual_Volatility'] = [
            self.annual_volatility(self.strategy_returns),
            self.annual_volatility(self.benchmark_returns)]
            #self.calculate_annual_volatility(self.excess_returns)]

        metrics['Max_Drawdown(%)'] = [
            self.simple_max_drawdown(self.strategy_returns),
            self.compound_max_drawdown(self.benchmark_returns)]
            #self.calculate_max_drawdown(self.excess_returns)]

        metrics['Sharpe'] = [
            self.sharpe_ratio(self.strategy_returns),
            self.sharpe_ratio(self.benchmark_returns, simple=False)]
            #self.calculate_sharpe_ratio(self.excess_returns)]

        metrics['Sortino'] = [
            self.sortino_ratio(self.strategy_returns),
            self.sortino_ratio(self.benchmark_returns, simple=False)]
            #self.calculate_sortino_ratio(self.excess_returns)]

        df = pd.DataFrame(metrics)
        df = df.round(4)
        df.set_index([['strategy', 'benchmark']], inplace=True)
        df.index.name = 'Backtest'
        df.reset_index(inplace=True)

        return df

    def plot_returns_distribution(self):
        """ """
        fig2 = plt.figure(figsize=(25, 6))
        plt.rcParams['font.size'] = 16
        plt.rcParams["figure.autolayout"] = True

        plt.subplot(1, 2, 2)
        plt.hist(self.benchmark_returns, bins=30, alpha=0.5, label='Benchmark Returns')
        plt.axvline(self.benchmark_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.benchmark_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        plt.twinx()
        kde = gaussian_kde(self.benchmark_returns.values)
        x1 = np.linspace(np.min(self.benchmark_returns.values), np.max(self.benchmark_returns.values), 100)
        mean1 = np.mean(self.benchmark_returns.values)
        std1 = np.std(self.benchmark_returns.values, ddof=1)
        y11 = kde(x1)
        y12 = norm.pdf(x1, mean1, std1)
        plt.plot(x1, y11, color='blue', label='Distribution')
        plt.plot(x1, y12, color='black', label='Normal Distribution')
        plt.title('Benchmark Returns Distribution')
        plt.legend()

        plt.subplot(1, 2, 1)
        plt.hist(self.strategy_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.strategy_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.strategy_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        plt.twinx()
        kde = gaussian_kde(self.strategy_returns.values)
        x2 = np.linspace(np.min(self.strategy_returns.values), np.max(self.strategy_returns.values), 100)
        mean2 = np.mean(self.strategy_returns.values)
        std2 = np.std(self.strategy_returns.values, ddof=1)
        y21 = kde(x2)
        y22 = norm.pdf(x2, mean2, std2)
        plt.plot(x2, y21, color='blue', label='Distribution')
        plt.plot(x1, y22, color='black', label='Normal Distribution')
        plt.title('Strategy Returns Distribution')
        plt.legend()
        """
        plt.subplot(1, 3, 3)
        plt.hist(self.excess_returns, bins=30, alpha=0.5, label='Strategy Returns')
        plt.axvline(self.excess_returns.mean(), color='r', linestyle='dashed', linewidth=1.5, label='Mean')
        plt.axvline(np.median(self.excess_returns), color='g', linestyle='dashed', linewidth=1.5, label='Median')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')

        plt.twinx()
        kde = gaussian_kde(self.excess_returns.values)
        x3 = np.linspace(np.min(self.excess_returns.values), np.max(self.excess_returns.values), 100)
        mean3 = np.mean(self.excess_returns.values)
        std3 = np.std(self.excess_returns.values, ddof=1)
        y31 = kde(x3)
        y32 = norm.pdf(x3, mean3, std3)
        plt.plot(x3, y31, color='blue', label='Distribution')
        plt.plot(x3, y32, color='black', label='Normal Distribution')
        plt.title('Excess Returns Distribution')
        plt.legend()
        """
        return fig2

    def return_stats(self):
        """"""
        values1 = self.strategy_returns.values
        values2 = self.benchmark_returns.values
        #values3 = self.excess_returns.values
        values_list = [values1, values2]

        df = pd.DataFrame()
        for value in values_list:
            stats = dict()

            stats['mean'] = np.round(np.mean(value), decimals=4)
            stats['std'] = np.round(np.std(value), decimals=4)
            stats['median'] = np.round(np.median(value), decimals=4)
            stats['kurtosis'] = np.round(kurtosis(value), decimals=4)
            stats['skewness'] = np.round(skew(value), decimals=4)

            df1 = pd.DataFrame(stats, index=[0])
            df = df._append(df1)

        df.set_index([['strategy', 'benchmark']], drop=True, inplace=True)
        df.index.name = 'Stats'
        df.reset_index(inplace=True)

        return df

    def run_backtest(self):
        """"""
        self.execute_trades()
        self.plot_cumulative_returns()

        with PdfPages(self.save_path) as pdf:
            fig1 = plt.figure(figsize=(25, 8))
            plt.rcParams["figure.autolayout"] = True
            gs = GridSpec(1, 2, figure=fig1, width_ratios=[3, 2])
            ax1_1 = fig1.add_subplot(gs[0, 0])
            ax1_1.axis('off')
            ax1_1.text(0.2, 0.5, self.description, fontsize=20, ha='center', va='center')
            ax1_2 = fig1.add_subplot(gs[0, 1])
            ax1_2.axis('off')
            table1 = ax1_2.table(cellText=self.get_portfolio_returns()[1].values,
                                 colLabels=self.get_portfolio_returns()[1].columns,
                                 cellLoc='center', loc='center')
            table1.auto_set_font_size(False)
            table1.set_fontsize(17)
            table1.scale(1, 2.5)
            pdf.savefig(fig1)

            pdf.savefig(self.plot_performance())
            pdf.savefig(self.plot_positions_bar())

            df1 = self.get_dataframe()
            fig2, ax2 = plt.subplots(figsize=(25, 5))
            plt.rcParams["figure.autolayout"] = True
            ax2.axis('off')  # Hide the axis
            table2 = ax2.table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
            table2.auto_set_font_size(True)
            table2.scale(1.5, 1.5)
            pdf.savefig(fig2)

            pdf.savefig(self.plot_returns_distribution())

            df2 = self.return_stats()
            fig4, ax4 = plt.subplots(figsize=(25, 5))
            plt.rcParams["figure.autolayout"] = True
            ax4.axis('off')  # Hide the axis
            table3 = ax4.table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
            table3.auto_set_font_size(True)
            table3.scale(1.5, 1.5)
            pdf.savefig(fig4)
