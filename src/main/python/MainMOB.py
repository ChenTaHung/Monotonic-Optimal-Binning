#%%
import pandas as pd
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python')
from MOB.MOB import MOB

if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
    df['default'] = df['default'] - 1
    MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = "default", exclude_value = None)
    MOB_ALGO.setBinningConstraints(max_bins = 6, min_bins = 4, 
                                   max_samples = 0.4, min_samples = 0.05, 
                                   min_bads = 0.05, 
                                   init_pvalue = 0.35)
    BinningResultTable = MOB_ALGO.runMOB(mergeMethod='MFB')
    print(BinningResultTable)
    
    MOB_ALGO.plotBinsSummary(binSummaryTable = BinningResultTable)
# %%
