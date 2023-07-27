#%%
import pandas as pd
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src')
from MOBPY.MOB import MOB
from MOBPY.plot.MOB_PLOT import MOB_PLOT
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
                
            # mergeMethod = 'Stats' means to run MOB algorithm under statistical base

            StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
            
            if x == 'Durationinmonth' :
                SizeBinning
                MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, figsavePath='/Users/chentahung/Desktop/git/mob-py/doc/charts/Durationinmonth-Size.png', dpi = 1200)
            elif x == 'Creditamount' :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, figsavePath='/Users/chentahung/Desktop/git/mob-py/doc/charts/Creditamount-Size.png', dpi = 1200)
                MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, figsavePath='/Users/chentahung/Desktop/git/mob-py/doc/charts/Creditamount-Stats.png', dpi = 1200)
            else :
                MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning)

            # print('Bins Size Base')
            # MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning)
            # print('Statisitcal Base')
            # MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning)


# %%

if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
    df['default'] = df['default'] - 1
    MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = None) 
    # Set Binning Constraints (Must-Do!)
    MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
    # mergeMethod = 'Size' means to run MOB algorithm under bins size base
    SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
    MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning)
    MOB_ALGO.applyMOB(df['Durationinmonth'])

# %%

if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance2.csv')
    MOB_ALGO = MOB(data = df, var = 'bmi', response = 'insuranceclaim', exclude_value = None) 
    # Set Binning Constraints (Must-Do!)
    MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
    # mergeMethod = 'Size' means to run MOB algorithm under bins size base
    StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
    MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = 'bmi')
# %%
