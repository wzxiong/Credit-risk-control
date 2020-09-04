#统计各个月份好坏人数

groups = df.groupby(['ym'])
for name,group in groups:
    print('{},{},{},{},{}'.format(name,group.shape[0],group[group['ever_m2plus']==1].shape[0],group[group['ever_m2plus']==0].shape[0],group['ever_m2plus'].mean()))

#统计各分段逾期率
def print_badrate(mobs):
    bins = range(450, 750,10)
    bins = [0] + bins
    bins.append(900)
    df['cutBins'] = pd.cut(df['fico'], bins)
    groups = df.groupby(['ym','cutBins'])
    diction = {'name':[],'badrate':[]}
    for name,group in groups:
        diction['name'].append(name)
        diction['badrate'].append(group['ever_m2plus'].mean())
    ddf = pd.DataFrame.from_dict(diction)
    ddf['month'] = [i[0] for i in ddf['name']]
    ddf['cuts'] = [i[1] for i in ddf['name']]
    test  = None
    for i in sorted(set(ddf['month'])):
        newtest = ddf[ddf['month'] == i][['cuts','badrate']]
        if test is not None:
            test = test.merge(newtest,left_on='cuts',right_on='cuts',how='left')
        else:
            test = newtest
    test.columns = ['cuts'] +[str(i) for i in sorted(set(ddf['month']))]
    #output = pd.DataFrame({'name':names,'badrate':badrates,'counts':counts})
    test.to_csv('badrates_'+mobs+'.csv',index=None)
    print(test)
print_badrate('mob12')

#统计各分段坏人数
def print_count(mobs):
    bins = range(450, 800,10)
    bins = [0] + bins
    bins.append(900)
    df['cutBins'] = pd.cut(df['fico'], bins)
    groups = df.groupby(['ym','cutBins'])
    diction = {'name':[],'badrate':[]}
    for name,group in groups:
        diction['name'].append(name)
        diction['badrate'].append(group.shape[0])
    ddf = pd.DataFrame.from_dict(diction)
    ddf['month'] = [i[0] for i in ddf['name']]
    ddf['cuts'] = [i[1] for i in ddf['name']]
    test  = None
    for i in sorted(set(ddf['month'])):
        newtest = ddf[ddf['month'] == i][['cuts','badrate']]
        if test is not None:
            test = test.merge(newtest,left_on='cuts',right_on='cuts',how='left')
        else:
            test = newtest
    test.columns = ['cuts'] +[str(i) for i in sorted(set(ddf['month']))]
    #output = pd.DataFrame({'name':names,'badrate':badrates,'counts':counts})
    test.to_csv('counts_'+mobs+'.csv',index=None)
    print(test)
