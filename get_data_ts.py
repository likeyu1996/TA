# -*- coding: utf-8 -*-
import tushare as ts

pro = ts.pro_api('6144f3417ac5da6235442a7bafe9ba6931c3fba8dcdd8946d089f862')

'''
def get_stock_daily(pool,start,end):
    for i in pool:
        df_qfq = ts.get_h_data(i, start=start, end=end, pause=4)
        print(df_qfq)
        df_nfq = ts.get_h_data(i, start=start, end=end, pause=4, autype=None)
        df_qfq.to_csv(path_or_buf='./data_ts/%s_qfq.csv' % i, index=0, encoding='gbk')
        df_nfq.to_csv(path_or_buf='./data_ts/%s_nfq.csv' % i, index=0, encoding='gbk')
'''
def get_stock_daily(pool,start,end):
    for i in pool:
        ts_code = i
        df = pro.query('daily',start_date=start, end_date=end, ts_code=ts_code,fields='trade_date,open,close,high,low',is_open='0')
        df.to_csv(path_or_buf='./data_ts/%s.csv' % ts_code, index=0)


def get_index_daily(pool,start,end):
    for i in pool:
        df = ts.get_hist_data(i, start=start, end=end,pause=4)
        # pd.to_datetime()
        df.to_csv(path_or_buf='./data_ts/%s.csv' % i, encoding='gbk')


stock_pool = ['600519.SH','601398.SH','000858.SZ']
index_pool = ['000300','000016','399905']
index_pool_2 = ['hs300','sz50','csi500']
get_stock_daily(stock_pool,'2018-01-01','2020-11-10')
# get_index_daily(index_pool_2,'2015-01-01','2020-11-10')

'''



def get_stock_price(pool,start,end):
    for i in pool:
        ts_code = i
        df = pro.query('daily',start_date=start, end_date=end, ts_code=ts_code,fields='trade_date,open,close,high,low',is_open='0')
        df.to_csv(path_or_buf='./price_data/%s.csv' % ts_code, index=0)


def get_index_price(pool,start,end):
    for i in pool:
        ts_code = i
        # df = pro.query('index_daily',start_date=start, end_date=end, ts_code=ts_code,fields='trade_date,open,close,high,low',is_open='0')
        df = ts.get_hist_data(ts_code,start=start,end=end)
        df.to_csv(path_or_buf='./price_data/%s.csv' % ts_code, index=0,encoding='gbk')
'''
