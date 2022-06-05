import numpy as np                                                      
import pandas as pd                                                       
from datetime import datetime as dt


import tushare as ts
stock_id = '131803'
df = ts.get_hist_data(stock_id)
print(df)
df.to_csv('{}-tushare-download.csv'.format(stock_id))


df = pd.read_csv('./{}-tushare-download.csv'.format(stock_id))

print(np.shape(df))
df.head()

'''股票数据的特征
*date：日期
*open：开盘价
*high：最高价
*close：收盘价
*low：最低价
*volume：成交量
*price_change：价格变动 - 收盘价只差
*p_change：涨跌幅 - 百分比，price_change/close (前日)
*ma5：5 日均价
*ma10：10 日均价
*ma20: 20 日均价
*v_ma5: 5 日均量
*v_ma10: 10 日均量
*v_ma20: 20 日均量
'''

# 将每一个数据的键值的类型从字符串转为日期

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
# 按照时间升序排列
df.sort_values(by=['date'], inplace=True, ascending=True)
df.tail()

# 检测是否有缺失数据 NaNs
df.dropna(axis=0, inplace=True)
df.isna().sum()



# pro
import numpy as np                                                      
import pandas as pd  

import tushare as ts
ts.set_token('b291020f2ce1feed95b8c59b97ae026d737877376bbceec4b424b65b')
pro = ts.pro_api()


import os
stocks = ['603236','600391','600079','600196','300751','002049','603160','002230','600588','000002']
for i in stocks:    
    filename = "{}-tushare-download.csv".format(i)
    if not os.path.isfile(filename):
        print(filename)
        #df = ts.get_hist_data(i)
        df = pro.daily(ts_code=i, start_date='20150101', end_date='20200808')
        df.to_csv(filename)
