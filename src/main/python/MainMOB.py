#%%
import pandas as pd
import numpy as np
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python')
from MOB.MOB import MOB

if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
    df['default'] = df['default'] - 1
    MOB_ALGO = MOB(data = df, var = 'Creditamount', response = "default", exclude_value = None)
    MOB_ALGO.setBinningConstraints(max_bins = 6, min_bins = 3, 
                                   max_samples = 0.4, min_samples = 0.05, 
                                   min_bads = 0.05, 
                                   init_pvalue = 0.35, 
                                   maximize_bins=False)
    
    SFB = MOB_ALGO.runMOB(mergeMethod='SFB')
    print(SFB)
    MOB_ALGO.plotBinsSummary(binSummaryTable = SFB)
    
    MFB = MOB_ALGO.runMOB(mergeMethod='MFB')
    print(MFB)
    MOB_ALGO.plotBinsSummary(binSummaryTable = MFB)
