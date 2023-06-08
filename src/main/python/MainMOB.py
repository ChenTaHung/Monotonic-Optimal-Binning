#%%
import pandas as pd
import numpy as np
import os
os.chdir(os.getcwd())
from MOB.MOB import MOB
#%%

df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
df['default'] = df['default'] - 1
for x in df.columns[df.dtypes != 'object'] :
    if x != 'default'  :
        MOB_ALGO = MOB(data = df, var = x, response = "default", exclude_value = None)
        MOB_ALGO.setBinningConstraints(max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
        print(f' ===================== {x} =====================')
        SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
        # print(SFB)
        MOB_ALGO.plotBinsSummary(binSummaryTable = SizeBinning)
        
        StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
        # print(MFB)
        MOB_ALGO.plotBinsSummary(binSummaryTable = StatsBinning)

    

'''
代辦：
Missing Data bads = 0 時候 woe 計算 (exc_value 同理）
'''

# %%
