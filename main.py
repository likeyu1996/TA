# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa import stattools
from statsmodels.tsa import arima_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_te
from scipy import stats


def data_clean(path):
    df = pd.read_csv(path,encoding='gbk')[['日期','收盘价']].sort_values(by='日期',ascending=False)
    df.columns = ['date','close']
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date']
    simple_return = np.array(df['close']/df['close'].shift(-1) - 1)
    log_return = np.array(np.log(df['close']/df['close'].shift(-1)))
    new_df = pd.DataFrame([simple_return,log_return],index=['simple_r','log_r']).T
    con_df = pd.concat([df,new_df],axis=1)
    con_df.drop(len(con_df)-1,inplace=True)
    return con_df


def index_week_clean(path):
    df = pd.read_csv(path,encoding='gbk')[['日期','收盘价']].sort_values(by='日期',ascending=False)
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date']
    df['adj_week'] = df['date']
    for i in range(df['date'].size):
        df.loc[i,'week'] = df.loc[i,'date'].isocalendar()[1]
    # 逆序计周数
    for i in range(df['week'].size-1, -1, -1):
        if i == df['week'].size-1:
            df.loc[i,'adj_week'] = 1
        else:
            if df.loc[i,'week'] != df.loc[i+1,'week']:
                df.loc[i,'adj_week'] = df.loc[i+1,'adj_week']+ 1
            else:
                df.loc[i,'adj_week'] = df.loc[i+1,'adj_week']
    adj_week_list = list(reversed(list(set(df['adj_week']))))
    new_close_list = []
    for i in adj_week_list:
        for j in range(df['adj_week'].size):
            if i == df.loc[j,'adj_week']:
                new_close_list.append(df.loc[j,'close'])
                break
    cache_df = pd.DataFrame([adj_week_list,new_close_list],index=['adj_week','close']).T
    log_return = np.array(np.log(cache_df['close']/cache_df['close'].shift(-1)))
    new_df = pd.DataFrame([log_return],index=['log_r']).T
    con_df = pd.concat([cache_df,new_df],axis=1)
    con_df.drop(len(con_df)-1,inplace=True)
    return con_df

path_folder = 'data_163'
# 1贵州茅台 工商银行 五粮液 日序列


path_gzmt = './'+path_folder+'/sh600519.csv'
path_icbc = './'+path_folder+'/sh601398.csv'
path_wly = './'+path_folder+'/sz000858.csv'

gzmt = data_clean(path_gzmt)
icbc = data_clean(path_icbc)
wly = data_clean(path_wly)

# 1.1简单收益率 对数收益率

gzmt_simple_return = np.array(gzmt['simple_r'][::-1])
icbc_simple_return = np.array(icbc['simple_r'][::-1])
wly_simple_return = np.array(wly['simple_r'][::-1])

gzmt_log_return = np.array(gzmt['log_r'][::-1])
icbc_log_return = np.array(icbc['log_r'][::-1])
wly_log_return = np.array(wly['log_r'][::-1])

# 1.2对数收益率时序图
'''
# sns.lineplot(x='date',y='log_r',data=gzmt_log_return)

plt.figure(figsize=(16, 9))
sns.lineplot(data=gzmt_log_return)
plt.savefig('./answer/gzmt_log_r.png')

plt.figure(figsize=(16, 9))
sns.lineplot(data=icbc_log_return)
plt.savefig('./answer/icbc_log_r.png')

plt.figure(figsize=(16, 9))
sns.lineplot(data=wly_log_return)
plt.savefig('./answer/wly_log_r.png')
'''
# 1.3检验数据平稳性
'''
gzmt_adf = stattools.adfuller(gzmt_log_return)
icbc_adf = stattools.adfuller(icbc_log_return)
wly_adf = stattools.adfuller(wly_log_return)
'''
# 1.4通过信息准则建立合适的模型
'''
# 自相关偏自相关图
gzmt_acf=stattools.acf(gzmt_log_return)
plot_acf(gzmt_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/gzmt_acf.png')
plot_pacf(gzmt_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/gzmt_pacf.png')

icbc_acf=stattools.acf(icbc_log_return)
plot_acf(icbc_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/icbc_acf.png')
plot_pacf(icbc_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/icbc_pacf.png')

wly_acf=stattools.acf(wly_log_return)
plot_acf(wly_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/wly_acf.png')
plot_pacf(wly_log_return,use_vlines=True,lags=60)
plt.savefig('./answer/wly_pacf.png')

# Ljung-Box检验(检验时间序列或者拟合模型的残差是否为白噪声序列)
gzmt_lb = lb_te(gzmt_log_return,lags=20,return_df=True)
icbc_lb = lb_te(icbc_log_return,lags=20,return_df=True)
wly_lb = lb_te(wly_log_return,lags=20,return_df=True)
lb_result = pd.concat([gzmt_lb,icbc_lb,wly_lb],axis=1,join='inner')
lb_result.to_csv(path_or_buf='./answer/lb_result.csv')
plt.figure(figsize=(16, 9))
sns.lineplot(data=gzmt_lb['lb_pvalue'])
plt.savefig('./answer/gzmt_lb.png')
plt.figure(figsize=(16, 9))
sns.lineplot(data=icbc_lb['lb_pvalue'])
plt.savefig('./answer/icbc_lb.png')
plt.figure(figsize=(16, 9))
sns.lineplot(data=wly_lb['lb_pvalue'])
plt.savefig('./answer/wly_lb.png')
'''

# AIC(赤池信息准则) BIC(贝叶斯信息准则)
ic_max_ar = 10
ic_max_ma = 2
gzmt_order_aic = stattools.arma_order_select_ic(gzmt_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='aic')
gzmt_order_bic = stattools.arma_order_select_ic(gzmt_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='bic')

icbc_order_aic = stattools.arma_order_select_ic(icbc_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='aic')
icbc_order_bic = stattools.arma_order_select_ic(icbc_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='bic')

wly_order_aic = stattools.arma_order_select_ic(wly_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='aic')
wly_order_bic = stattools.arma_order_select_ic(wly_log_return,max_ar=ic_max_ar,max_ma=ic_max_ma,ic='bic')

print(gzmt_order_aic,gzmt_order_bic,icbc_order_aic,icbc_order_bic,wly_order_aic,wly_order_bic)

# 1.5模型拟合 参数估计并进行合意性诊断
'''
gzmt_arma = arima_model.ARMA(gzmt_log_return,(6,1)).fit()
icbc_arma = arima_model.ARMA(icbc_log_return,(6,2)).fit()
wly_arma = arima_model.ARMA(wly_log_return,(0,1)).fit()
print(gzmt_arma.summary())
print(icbc_arma.summary())
print(wly_arma.summary())
'''
# 1.6预测未来十天的数据并给出预测方差
'''

'''
# 2沪深300 上证50 中证500 周序列
'''
path_hs300 = './'+path_folder+'/sh000300.csv'
path_sz50 = './'+path_folder+'/sh000016.csv'
path_csi500 = './'+path_folder+'/sh000905.csv'
index_week_clean(path_hs300).to_csv('./'+path_folder+'/hs300.csv',index=False)
index_week_clean(path_sz50).to_csv('./'+path_folder+'/sz50.csv',index=False)
index_week_clean(path_csi500).to_csv('./'+path_folder+'/csi500.csv',index=False)
'''
'''
hs300 = pd.read_csv('./'+path_folder+'/hs300.csv')
sz50 = pd.read_csv('./'+path_folder+'/sz50.csv')
csi500 = pd.read_csv('./'+path_folder+'/csi500.csv')
hs300_log_r = np.array(hs300['log_r'][::-1])
sz50_log_r = np.array(sz50['log_r'][::-1])
csi500_log_r = np.array(csi500['log_r'][::-1])
'''
# 四种方法截然不同
'''
print(np.array(hs300_log_r))
print(np.array(hs300_log_r[::-1]))
print(np.array(hs300_log_r.sort_values(ascending=False)))
print(np.array(hs300_log_r.rank(ascending=False)))
'''
# 2.1计算对数收益率序列的均值标准差偏度和超额峰度
'''
hs300_mean = np.mean(hs300_log_r)
sz50_mean = np.mean(sz50_log_r)
csi500_mean = np.mean(csi500_log_r)
hs300_std = np.std(hs300_log_r)
sz50_std = np.std(sz50_log_r)
csi500_std = np.std(csi500_log_r)
hs300_skew = stats.skew(hs300_log_r)
sz50_skew = stats.skew(sz50_log_r)
csi500_skew = stats.skew(csi500_log_r)
hs300_kurtosis = stats.kurtosis(hs300_log_r)
sz50_kurtosis = stats.kurtosis(sz50_log_r)
csi500_kurtosis = stats.kurtosis(csi500_log_r)
dic_2_1 = {'hs300':[hs300_mean,hs300_std,hs300_skew,hs300_kurtosis,hs300_kurtosis-3],
           'sz50':[sz50_mean,sz50_std,sz50_skew,sz50_kurtosis,sz50_kurtosis-3],
           'csi500':[csi500_mean,csi500_std,csi500_skew,csi500_kurtosis,csi500_kurtosis-3]
           }
df_2_1 = pd.DataFrame(dic_2_1,index=['mean','std','skew','kurtosis','excess_k']).T
df_2_1.to_csv('./answer/answer_2_1.csv')
'''
# 2.2比较三者的经验分布 判断是否属于正态分布
'''
plt.figure(figsize=(16, 9))
sns.distplot(hs300_log_r)
plt.savefig('./answer/hs300_log_r.png')
plt.figure(figsize=(16, 9))
sns.distplot(sz50_log_r)
plt.savefig('./answer/sz50_log_r.png')
plt.figure(figsize=(16, 9))
sns.distplot(csi500_log_r)
plt.savefig('./answer/csi500_log_r.png')

# 改进的KS检验Anderson-Darling 原假设H0为样本服从特定分布
hs300_ks = stats.anderson(hs300_log_r,'norm')
sz50_ks = stats.anderson(sz50_log_r,'norm')
csi500_ks = stats.anderson(csi500_log_r,'norm')
print(hs300_ks)
print(sz50_ks)
print(csi500_ks)
'''
# 2.3时序图 判断领先滞后关系
'''
plt.figure(figsize=(16, 9))
sns.lineplot(data=hs300_log_r)
plt.savefig('./answer/hs300_log_r_2_3.png')
plt.figure(figsize=(16, 9))
sns.lineplot(data=sz50_log_r)
plt.savefig('./answer/sz50_log_r_2_3.png')
plt.figure(figsize=(16, 9))
sns.lineplot(data=csi500_log_r)
plt.savefig('./answer/csi500_log_r_2_3.png')
plt.figure(figsize=(64, 27))
data_cache = pd.DataFrame([hs300_log_r,sz50_log_r,csi500_log_r],index=['hs300','sz50','csi500']).T
sns.lineplot(data=data_cache)
plt.savefig('./answer/log_r_2_3.png')

# Engel-Granger EG协整检验 原假设H0为不存在协整
print('hs300&sz50',stattools.coint(hs300_log_r, sz50_log_r))
print('sz50&csi500',stattools.coint(sz50_log_r, csi500_log_r))
print('csi500&hs300',stattools.coint(csi500_log_r,hs300_log_r))
# 全都拒绝原假设 因此三者两两之间存在协整关系 可以回归
# 
'''
# 2.4平稳性检验 构建指数间的预测模型
'''
# ADF检验原假设为存在单位根
hs300_adf = stattools.adfuller(hs300_log_r)
sz50_adf = stattools.adfuller(sz50_log_r)
csi500_adf = stattools.adfuller(csi500_log_r)
print(hs300_adf)
print(sz50_adf)
print(csi500_adf)
'''
# 2.5对3 4的结论进行经济学或投资逻辑上的解释
'''
沪深300是整个市场的晴雨表
上证50是优质大盘股的代表
中证500是小盘股的代表
'''
# 2.6运用模型对未来三周的数据进行预测
