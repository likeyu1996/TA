# -*- coding: utf-8 -*-
import datetime
import requests
import urllib
'''
TAG	意义
TCLOSE	收盘价
HIGH	最高价
LOW	最低价
CHG	涨跌额
PCHG	涨跌幅
TURNOVER	换手率
VOTURNOVER	成交量
VATURNOVER	成交金额
TCAP	总市值
MCAP	流通市值
'''

def get_stock_163_history(pool,start,end):
    # 根据网易规则进行股票编号转换
    for code in pool:
        org_code = code
        url_code = code.replace("sh", "0").replace("sz", "1")
        url = 'http://quotes.money.163.com/service/chddata.html?code='+url_code+'&start='+start+'&end='+end+'&fields=TCLOSE;HIGH;LOW'
        urllib.request.urlretrieve(url, "./data_163/%s.csv" % org_code)


start_1 = datetime.datetime(2018, 1, 1).strftime("%Y%m%d")
start_2 = datetime.datetime(2015, 1, 1).strftime("%Y%m%d")
end = datetime.date.today().strftime("%Y%m%d")
stock_pool = ['sh600519', 'sh601398', 'sz000858']
index_pool = ['sh000300', 'sh000016', 'sh000905']
get_stock_163_history(stock_pool,start_1,end)
get_stock_163_history(index_pool,start_2,end)

'''
    iget = 0
    r = requests.get(url)
    line=str(r.content)
    print(line)
    #指定抓取到存储的文件名，每个个股一个文件
    filename='../data/'+org_code+'.his'
    fout=open(filename,"w")

    # 解析抓取回来的数据，格式化存储
    lines = line.split("\\n")
    for i in range(1,len(lines)):
        lines[i]=lines[i].replace("\\r","")
        res=lines[i].split(",")
        #如果列数不足，则此行格式有误，丢弃
        if len(res)>11:
            fout.write(res[0])
            for j in range(3,len(res)):
                fout.write("\t"+res[j])
            fout.write("\n")
            fout.flush()
            iget=iget+1
    fout.close
    return iget
'''
