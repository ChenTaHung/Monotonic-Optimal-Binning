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
            print('Bins Size Base')
            MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning)
                
            # mergeMethod = 'Stats' means to run MOB algorithm under statistical base
            StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')
            print('Statisitcal Base')
            MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning)


# %%

if __name__ == '__main__' :
    df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
    df['default'] = df['default'] - 1
    MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = [11,15]) 
    # Set Binning Constraints (Must-Do!)
    MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
    # mergeMethod = 'Size' means to run MOB algorithm under bins size base
    SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
    MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')
    

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
