#%%
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance3r2.csv')

df
#%%

df_grp = df.groupby('age')['insuranceclaim'].agg(['count', 'sum', 'std', 'max', 'min']).reset_index()
df_grp
# %%

# assign fake assign values and metric 
df_grp_sum = df_grp.copy()
df_grp_sum.loc[(df_grp_sum['age'] <=39), 'assignValue'] = 39
df_grp_sum.loc[(df_grp_sum['age'] > 39) & (df_grp_sum['age'] <= 55), 'assignValue'] = 55
df_grp_sum.loc[(df_grp_sum['age'] > 55) & (df_grp_sum['age'] <= 61), 'assignValue'] = 61
df_grp_sum.loc[(df_grp_sum['age'] > 61), 'assignValue'] = 64

df_grp_sum['assignMetric'] = df_grp_sum['assignValue'].map({39:342, 55:287, 61:102, 64:52})
df_grp_sum.sort_values(by = 'age', inplace = True)
df_grp_sum
# %%
df_grp_sum.groupby('assignValue')['insuranceclaim'].cumsum()
# %%
df_grp = df.groupby('age')['insuranceclaim'].std().reset_index()
df_grp
df_grp_std = df_grp.copy()

df_grp_std.loc[(df_grp_std['age'] <=19), 'assignValue'] = 19
df_grp_std.loc[(df_grp_std['age'] > 19), 'assignValue'] = 64

df_grp_std['assignMetric'] = df_grp_std['assignValue'].map({19:0.432095, 64:0.379143})
df_grp_std.sort_values(by = 'age', inplace = True)
df_grp_std


# %%
#std
df_grp_std.groupby('assignValue')['insuranceclaim'].expanding().std()
# %%
_df = df_grp.copy()
_df.loc[(_df['age'] <=19), 'assignValue'] = 19
_df.loc[(_df['age'] > 19), 'assignValue'] = 64
_df
# %%
df_grp_std = df_grp.copy()
df_grp_std.loc[(df_grp_std['age'] <=19), 'assignValue'] = 18
df_grp_std.loc[(df_grp_std['age'] > 19), 'assignValue'] = 20

df_grp_std['assignMetric'] = df_grp_std['assignValue'].map({18:0.432095, 20:0.379143})
df_grp_std

#%%
'''
        if (curcount + nextCount) == 2 :
            newStd = np.std(np.array(curmean, (nextSum/nextCount))) # newMean = newSum / newCount
        else :
            newStd = np.sqrt(((curcount * (curstd ** 2)) + (nextSum * (nextStd ** 2))) / (curcount + nextCount - 1))

'''

df_grp_std['mean'] = df_grp_std['sum'] / df_grp_std['count']


cumStd = 0
cumStdList = []
for curId in range(1,len(df_grp_std)) :
    '''
    ['age', 'count', 'sum', 'std', 'max', 'min', 'assignValue', 'assignMetric']
    '''
    curvar = df_grp_std.loc[curId, 'var']
    assignvar = df_grp_std.loc[curId, 'assignValue']
    
    if curvar == assignvar : # first record of the bin
        cumStd = df_grp_std.loc[curId, 'std']
        cumStdList.append(cumStd)
        cumStd = 0
    else :
        curcount = df_grp_std.loc[curId, 'count']
        nextcount = df_grp_std.loc[curId + 1, 'count']

        curmean = df_grp_std.loc[curId, 'mean']
        nextmean = df_grp_std.loc[curId + 1, 'mean']
        
        curstd = df_grp_std.loc[curId, 'std']
        nextstd = df_grp_std.loc[curId + 1, 'std']
        
        nextsum = df_grp_std.loc[curId + 1, 'sum']
        if (curcount + nextcount) == 2 :
            cumStd = np.std(np.array(curmean, nextmean)) 
        else :
            cumStd = np.sqrt(((curcount * (cumStd ** 2)) + (nextsum * (nextstd ** 2))) / (curcount + nextcount - 1))
        
        
print(cumStdList)
        
        

    
    
        