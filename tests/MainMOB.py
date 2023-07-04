#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python')
from MOB.MOB import MOB
from MOB.plot.MOB_PLOT import MOB_PLOT
#%%
if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
    df['default'] = df['default'] - 1
    # A simple sample to generate the MOB result for numeric columns in the dataset.
    for x in df.columns[df.dtypes != 'object'] :
        if x != 'default'  : # preclude the default column.
            # Construct the object
            MOB_ALGO = MOB(data = df, var = x, response = 'default', exclude_value = 0) 
            # Set Binning Constraints (Must-Do!)
            MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                            max_samples = 0.4, min_samples = 0.05, 
                                            min_bads = 0.05, 
                                            init_pvalue = 0.4, 
                                            maximize_bins=True)
            print(f' ===================== {x} =====================')
            # mergeMethod = 'Size' means to run MOB algorithm under bins size base
            SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
            print('Bins Size Base')
            if x in ['Durationinmonth', 'Creditamount'] :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = x, 
                                         figsavePath = f'/Users/chentahung/Desktop/git/mob-py/doc/charts/{x}-Size.png', dpi = 800)
            else :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = x)
                
            # mergeMethod = 'Stats' means to run MOB algorithm under statistical base
            StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
            print('Statisitcal Base')
            if x == 'Creditamount' :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = x, 
                                         figsavePath = f'/Users/chentahung/Desktop/git/mob-py/doc/charts/{x}-Stats.png', dpi = 800)
            else :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = x)


# %%

if __name__ == '__main__' :
    MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = None) 
    # Set Binning Constraints (Must-Do!)
    MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
    # mergeMethod = 'Size' means to run MOB algorithm under bins size base
    SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
    
    StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
    
