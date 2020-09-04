import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import math
from itertools import combinations

def bin_frequency(x,y,n=10,bucket=[]):
    total = y.count()
    bad = y.sum()
    good = y.count()-y.sum()
    if not bucket:
        bucket=pd.qcut(x,n,duplicates='drop')
    d1 = pd.DataFrame({'x':x,'y':y,'bucket':bucket})
    d2 = d1.groupby('bucket',as_index=True)
    d3 = pd.DataFrame(d2.x.min(),columns=['min_bin'])
    d3['min_bin'] = d2.x.min()
    d3['max_bin'] = d2.x.max()
    d3['bad'] = d2.y.sum()
    d3['total'] = d2.y.count()
    d3['bad_rate'] = d3['bad']/d3['total']
    d3['badattr'] = d3['bad']/bad
    d3['goodattr'] = (d3['total'] - d3['bad'])/good
    d3['woe'] = np.log(d3['goodattr']/d3['badattr'])
    iv = ((d3['goodattr']-d3['badattr'])*d3['woe']).sum()
    d4 = (d3.sort_values(by='min_bin')).reset_index(drop=True)
    cut = []
    cut.append(float('-inf'))
    for i in d4.min_bin:
        cut.append(i)
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return iv
    
data_list = [train,df201901,df201902,df201903,df201904,df201905,df201906,df201907,df201908,df201909,df201910,df201911,df201912,df202001,df202002,df202003,df202004,df202005]
index_list = ['train','201901','201902','201903','201904','201905','201906','201907','201908','201909','201910','201911','201912','202001','202002','202003','202004','202005']

out = pd.DataFrame()
diction = defaultdict(list)
print('distribution')
for data in [train,df201901,df201902,df201903,df201904,df201905,df201906,df201907]:
    print(data.shape[0],data[data.label==1].shape[0],data[data.label==0].shape[0],data['label'].mean())
    for col in youshu_list:
        diction[col].append(bin_frequency(data[col],data['label']))

ddf = pd.DataFrame.from_dict(diction)
ddf.index=['train','201901','201902','201903','201904','201905','201906','201907']
out = pd.concat([out,ddf])
print('iv done')
print(out)

for data in data_list:
    for col in youshu_list:
        diction[col].append(data[col].describe().loc['mean'])

ddf = pd.DataFrame.from_dict(diction)
ddf.index = index_list
out = pd.concat([out,ddf])
print('mean done')

diction = defaultdict(list)
for data in data_list:
    for col in youshu_list:
        diction[col].append(data[col].describe().loc['std'])

ddf = pd.DataFrame.from_dict(diction)
ddf.index = index_list
out = pd.concat([out,ddf])
print('std done')

diction = defaultdict(list)
for data in data_list:
    for col in youshu_list:
        diction[col].append(data[col].describe().loc['25%'])

ddf = pd.DataFrame.from_dict(diction)
ddf.index = index_list
out = pd.concat([out,ddf])
print('25 done')

