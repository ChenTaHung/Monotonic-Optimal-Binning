#%%
import pandas as pd
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src')
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT

df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance2.csv')


#%%
P = PAVA(data = df, var = 'age', response = 'charges', metric='range', 
         add_var_aggFunc={'bmi':'mean', 'smoker':'sum', 'region':['max', 'min']})
P.runPAVA(sign = 'auto')
# %%
# print(P.orgDataAssignment)
# print(P.CSD_Summary)
# print(P.GCM_Summary)
print(P.PAV_Summary)
# %%
MOB_PLOT.plotPAVA_CSD(CSD_Summary = P.CSD_Summary)
# %%
MOB_PLOT.plotPAVA_CSD(CSD_Summary = P.CSD_Summary.drop_duplicates('age', keep = 'last'))
# %%
_res = P.applyPAVA(df['age'], 'interval')
# %%
for i in range(len(df)) :
    print(f'{df.iloc[i, 0]}   ----   {_res[i]}')
# %%
