import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import sys

def cal_ks(y_true, y_prob, n_bins=20):
    percentile = np.linspace(0, 100, n_bins + 1).tolist()
    bins = [np.percentile(y_prob, i) for i in percentile]
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    binids = np.digitize(y_prob, bins) - 1
    y_1 = sum(y_true == 1)
    y_0 = sum(y_true == 0)
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    bin_false = bin_total - bin_true
    true_pdf = bin_true / y_1
    false_pdf = bin_false / y_0
    true_cdf = np.cumsum(true_pdf)
    false_cdf = np.cumsum(false_pdf)
    ks_list = np.abs(true_cdf - false_cdf).tolist()
    ks = max(ks_list)
    return ks, ks_list, true_cdf,false_cdf

df = pd.read_csv(data)
train = df[df.dates < 20190000]
oot = df[df.dates > 20190000]

#ks 曲线
print('oot')
ks, ks_list, true_cdf,false_cdf = cal_ks(oot['ever_m2plus'],oot['pred'])
print(',trud_cdf,false_cdf,ks')
for a,i,j,k in zip(range(0,100,5),true_cdf[::-1],false_cdf[::-1],ks_list[::-1]):
    print(str(a)+'% '+str(1-i)+' '+str(1-j)+' '+str(k))
    
#抓坏率和坏人率
groups = df.groupby(['mob','ym'])
diction = {'name':[],'10%':[],'20%':[],'bad 10%':[],'bad 20%':[]}
for name,group in groups:
    if group.shape[0] < 100:
        continue
    diction['name'].append(name[1])
    #print(group.pred.quantile(0.1))
    diction['10%'].append(group[(group.pred > group.pred.quantile(0.9))&(group.ever_m2plus==1)].shape[0]*1.0/group[group.ever_m2plus==1].shape[0])
    diction['20%'].append(group[(group.pred > group.pred.quantile(0.8))&(group.ever_m2plus==1)].shape[0]*1.0/group[group.ever_m2plus==1].shape[0])
    diction['bad 10%'].append(group[group.pred > group.pred.quantile(0.9)]['ever_m2plus'].mean())
    diction['bad 20%'].append(group[group.pred > group.pred.quantile(0.8)]['ever_m2plus'].mean())

ddf = pd.DataFrame.from_dict(diction)
print(ddf)
