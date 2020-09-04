
def seg_fico(f,seg_list=[500,550,590]):
    ret = []
    for fico in f:
        if fico < seg_list[0]:
            ret.append('D')
        elif fico < seg_list[1] and fico >= seg_list[0]:
            ret.append('C')
        elif fico < seg_list[2] and fico >= seg_list[1]:
            ret.append('B')
        else:
            ret.append('A')
    return ret
    
A9['A9_seg'] = seg_fico(A9['fico'].values)
A10['A10_seg'] = seg_fico(A10['fico'].values,seg_list=[492,542,579])

A9 = A9[['id','date','A9_seg']]
A10 = A10[['id','date','label','A10_seg']]
A10 = A10.merge(A9,on=['id','date'],how='inner')
out = pd.DataFrame()
print(pd.crosstab(A10.A9_seg,A10.A10_seg))
out = pd.concat([out,pd.crosstab(A10.A9_seg,A10.A10_seg)])

A10bad = A10[A10.ever_m2plus==1]
print(pd.crosstab(A10bad.A9_seg,A10bad.A10_seg))
out = pd.concat([out,pd.crosstab(A10bad.A9_seg,A10bad.A10_seg)])

print(pd.crosstab(A10.A9_seg,A10.A10_seg,values=A10.ever_m2plus,aggfunc=np.average))
out = pd.concat([out,pd.crosstab(A10.A9_seg,A10.A10_seg,values=A10.ever_m2plus,aggfunc=np.average)])
