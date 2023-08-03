#%%
import pandas as pd
import numpy as np
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src')
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT

df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance2.csv')


#%%
# {'bmi':'mean', 'smoker':'sum', 'region':['max', 'min'], 'charges' : np.ptp}
P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc = None)
P.runPAVA()
# %%
# print(P.OrgDataAssignment)
print(P.CSD_Summary)
# print(P.GCM_Summary)
print(P.PAV_Summary)
# %%
MOB_PLOT.plotPAVACsd(CSD_Summary = P.CSD_Summary, figsavePath='/Users/chentahung/Desktop/git/mob-py/doc/charts/age-insuranceclaim-PAVA.png', dpi = 1200)

# %%
_res = P.applyPAVA(df['age'], 'interval')
# %%
for i in range(len(df)) :
    print(f'{df.iloc[i, 0]}   ----   {_res[i]}')
# %%
