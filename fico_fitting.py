import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def logit(x):
    return np.log(x/(1-x))
def cal_fico(score_trans):
    fico = 400 - (np.floor(50.5 * logit(score_trans)) + 0.5).astype(int)
    fico_min_val = 230
    fico_max_val = 890
    if fico > fico_max_val:
        fico = fico_max_val
    if fico < fico_min_val:
        fico = fico_min_val
    return fico

# 读取文件，需要有打分和label
new_df = pd.read_csv('oot1.csv')


new_df['logit_score'] = [logit(i) for i in new_df['pred']]
idx_bin = new_df.logit_score.quantile(np.arange(0.01, 1, 0.01))
new_df['cutBins2'] = pd.cut(new_df.logit_score, idx_bin)
true_odu = new_df.groupby('cutBins2')['label'].mean()
logit_true_odu = [logit(i) for i in true_odu]
logit_pred_odu = new_df.groupby('cutBins2')['logit_score'].mean()
paras = np.polyfit(logit_pred_odu[14:], logit_true_odu[14:], 2)
# paras为参数
print(paras)

# 画图看拟合程度
ss = np.arange(-6.5, 1, 0.1)
y_ss = ss * ss * paras[0] + ss * paras[1] + paras[2]
plt.plot(logit_pred_odu, logit_true_odu, '*')
plt.plot(ss, y_ss, 'k')

# 最好有个oot看各段情况
dfall = pd.read_csv('oot.csv')
dfall['logit_score'] = [logit(i) for i in dfall['pred']]
dfall['score_trans'] = [sigmoid(i) for i in np.array(dfall.logit_score **2 * paras[0] + dfall.logit_score * paras[1] + paras[2])]
dfall['fico'] = [cal_fico(i) for i in dfall['score_trans']]
fico_cut = list(range(290, 900, 10))
dfall['fico_bin'] = pd.cut(dfall.fico, fico_cut)
stats2 = dfall.groupby('fico_bin')[['label', 'score_trans']].mean()
stats2.columns = ['true_odu', 'theo_odu']
stats2['cnt'] = dfall.groupby('fico_bin')['label'].count()
stats2['n_odu'] = dfall.groupby('fico_bin')['label'].sum()
print(stats2)

