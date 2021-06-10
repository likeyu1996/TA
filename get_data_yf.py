# -*- coding: utf-8 -*-
import pandas_datareader.data as web
import datetime
import yfinance as yf
yf.pdr_override()


def get_stock_data(pool,start,end):
    for i in pool:
        df = web.get_data_yahoo(i,start,end)
        df.to_csv(path_or_buf='./data_yf/%s.csv' % i, encoding='gbk')


start_1=datetime.datetime(2018, 1, 1)
start_2=datetime.datetime(2015, 1, 1)
end=datetime.datetime.today()
# yahoo finance 沪市用SS代替 而非SH
stock_pool = ['600519.SS', '601398.SS', '000858.SZ']
index_pool = ['000300.SS', '000016.SS', '399905.SZ']
get_stock_data(stock_pool,start_1,end)
get_stock_data(index_pool,start_2,end)

