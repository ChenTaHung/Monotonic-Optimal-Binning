import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MOB.numeric.Monotone import Monotone
from MOB.numeric.OptimalBinning import OptimalBinning

class MOB:
    def __init__(self, data, var, response, exclude_value = None) :
        self.data = data
        self.var = var
        self.response = response 
        self.exclude_value = exclude_value
        '''
        _isNaExist : check Missing Existance
        _isExcValueExist : check Exclude Value Existance
        
        [attributes]
        self.df_missing : missing data subset
        self.df_excvalue : exclude value data subset
        self.df_sel : selected data subset
        
        '''
        if self.data[self.var].isna().sum() > 0 :
            _isNaExist = True
        else :
            _isNaExist = False
        
        # check exclude values exist
        if exclude_value != None :
            _isExcValueExist = True
        else :
            _isExcValueExist = False
        
        if _isNaExist & _isExcValueExist :
            self.df_missing = self.data.loc[self.data[self.var].isna(), :]
            
            if isinstance(self.exclude_value, list) :
                self.df_excvalue = self.data.loc[self.data[self.var].isin(exclude_value), :]
            elif isinstance(exclude_value, (float, int)) :
                self.df_excvalue = self.data.loc[self.data[self.var] == exclude_value, :]
                
            self.df_sel = self.data.loc[(self.data[self.var].notnull()) & (self.data[self.var] != exclude_value)]
            
        elif _isNaExist & ~_isExcValueExist: #only contain missing
            self.df_missing = self.data.loc[self.data[self.var].isna(), :]
            self.df_sel = self.data.loc[self.data[self.var].notnull()]
            
        elif ~_isNaExist & _isExcValueExist : #only contain exclude condition
            if isinstance(exclude_value, list) :
                self.df_excvalue = self.data.loc[self.data[self.var].isin(exclude_value), :]
            elif isinstance(exclude_value, (float, int)) :
                self.df_excvalue = self.data.loc[self.data[self.var] == exclude_value, :]
                
            self.df_sel = self.data.loc[self.data[self.var] != exclude_value]
        else:
            self.df_sel = self.data
            
        self.isNaExist = _isNaExist
        self.isExcValueExist = _isExcValueExist
    
    def setBinningConstraints(self, max_bins :int = 6, min_bins :int = 4, max_samples = 0.4, min_samples = 0.05, min_bads = 0.05, init_pvalue: float = 0.4, maximize_bins :bool = True) -> None:
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.init_pvalue = init_pvalue
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.min_bads = min_bads
        self.maximize_bins = maximize_bins

           
    def _summarizeBins(self, FinalOptTable):
        
        FinalOptTable = FinalOptTable[['start', 'end', 'total', 'bads', 'mean']].rename(columns = {'total': 'nsamples', 'bad' : 'bads', 'mean' : 'bad_rate'})
        FinalOptTable['dist_obs'] = FinalOptTable['nsamples'] / FinalOptTable['nsamples'].sum()
        FinalOptTable['dist_bads'] = FinalOptTable['bads'] / FinalOptTable['bads'].sum()
        FinalOptTable['goods'] = FinalOptTable['nsamples'] - FinalOptTable['bads']
        FinalOptTable['dist_goods'] = FinalOptTable['goods'] / FinalOptTable['goods'].sum()
        FinalOptTable['woe'] = np.log(FinalOptTable['dist_goods']/FinalOptTable['dist_bads'])
        FinalOptTable['iv_grp'] = (FinalOptTable['dist_goods'] - FinalOptTable['dist_bads']) * FinalOptTable['woe']
        
        return FinalOptTable      
  
    
    def runMOB(self, mergeMethod, sign = 'auto') :
        # Monotone
        MonotoneTuner = Monotone(data = self.df_sel, var = self.var, response = self.response)
        MonoTable = MonotoneTuner.tuneMonotone(sign = sign)
        self.MonoTable = MonoTable
        # Binning
        OptimalBinningMerger = OptimalBinning(resMonotoneTable = MonoTable, 
                                              max_bins = self.max_bins, min_bins = self.min_bins, 
                                              max_samples = self.max_samples, min_samples = self.min_samples, 
                                              min_bads = self.min_bads, init_pvalue = self.init_pvalue,
                                              maximize_bins = self.maximize_bins)
        
        finishBinningTable = OptimalBinningMerger.monoOptBinning(mergeMethod = mergeMethod)
        self.finishBinningTable = finishBinningTable
        # Summary
        if self.isNaExist and self.isExcValueExist : #contains missing and exclude value
            
            missingDF = pd.DataFrame({
                'start' : ['Missing'],
                'end' : ['Missing'],
                'total' : [len(self.df_missing)],
                'bads' : [self.df_missing[self.response].sum()],
                'mean' : [(self.df_missing[self.response].sum()) / (len(self.df_missing))]})
                
            excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename({'count':'total', 'sum':'bads'})
            excludeValueDF['mean'] = excludeValueDF['bads'] / excludeValueDF['total']
            
            completeBinningTable = pd.concat([finishBinningTable, missingDF, excludeValueDF], axis = 0, ignore_index = True)
        elif self.isNaExist & ~self.isExcValueExist : # contains missing but no special values
            missingDF = pd.DataFrame({
                'start' : ['Missing'],
                'end' : ['Missing'],
                'total' : [len(self.df_missing)],
                'bads' : [self.df_missing[self.response].sum()],
                'mean' : [(self.df_missing[self.response].sum()) / (len(self.df_missing))]})
            
            completeBinningTable = pd.concat([finishBinningTable, missingDF], axis = 0, ignore_index = True)
        elif ~self.isNaExist & self.isExcValueExist : # contains special values but no missing data
            excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename({'count':'total', 'sum':'bads'})
            excludeValueDF['mean'] = excludeValueDF['bads'] / excludeValueDF['total']
            
            completeBinningTable = pd.concat([finishBinningTable, excludeValueDF], axis = 0, ignore_index = True)
        else : # clean data with no missing and special values
            completeBinningTable = finishBinningTable
            
        outputTable = self._summarizeBins(FinalOptTable = completeBinningTable)
        
        return outputTable
            
            
    def plotBinsSummary(self, binSummaryTable, bar_fill = 'skyblue', bar_alpha = 0.5, bar_width = 0.5, bar_text_color = 'darkblue', 
                        line_color = 'orange', line_width = 3, dot_color = 'red', dot_size = 80, annotation_font_weight = 'bold'):
        
        fig, ax1 = plt.subplots(1,1,figsize = (12,8))
        
        binSummaryTable['end'] = pd.Categorical(binSummaryTable['end'])

        # Plot bar chart for 'dist_obs'
        bars = ax1.bar(np.arange(len(binSummaryTable['end'])), binSummaryTable['woe'], color = bar_fill, alpha = bar_alpha, width = bar_width)
        ax1.set_xticks(ticks = np.arange(len(binSummaryTable['end'])), labels = binSummaryTable['end'])
        ax1.axhline(0)
        ax1.set_xlabel('Interval End Value')
        ax1.set_ylabel('WoE', color = bar_text_color)

        # Add text
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height >= 0 :
                ax1.annotate(f'{binSummaryTable["dist_obs"].iloc[i]:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 10), textcoords='offset points', ha='center', va='top', weight = annotation_font_weight, c = bar_text_color)
            else :
                ax1.annotate(f'{binSummaryTable["dist_obs"].iloc[i]:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, -8), textcoords='offset points', ha='center', va='top', weight = annotation_font_weight, c = bar_text_color)
        ax2 = ax1.twinx()

        # Plot line chart for 'bad_rate'
        ax2.plot(np.arange(len(binSummaryTable['end'])), binSummaryTable['bad_rate'], color=line_color, label='Bad Rate', linewidth = line_width)
        ax2.scatter(np.arange(len(binSummaryTable['end'])), binSummaryTable['bad_rate'], color=dot_color, s = dot_size)
        ax2.set_xticks(ticks = np.arange(len(binSummaryTable['end'])), labels = binSummaryTable['end'])
        ax2.set_ylabel('Bad Rate', color=dot_color)
        
        # Add text
        med = binSummaryTable['bad_rate'].median()
        if binSummaryTable.iloc[-1, 4] - binSummaryTable.iloc[0, 4] > 0 :
            for i, val in enumerate(binSummaryTable['bad_rate']):
                if val <= med :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, -7.5), textcoords='offset points', ha='left', va='top', weight = annotation_font_weight, c = dot_color)
                else :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, 7.5), textcoords='offset points', ha='right', va='bottom', weight = annotation_font_weight, c = dot_color)
        else :
            for i, val in enumerate(binSummaryTable['bad_rate']):
                if val <= med :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, -7.5), textcoords='offset points', ha='right', va='top', weight = annotation_font_weight, c = dot_color)
                else :
                    ax2.annotate(f'{val:.1%}', xy=(i, val), xytext=(0, 7.5), textcoords='offset points', ha='left', va='bottom', weight = annotation_font_weight, c = dot_color)
        # Set Legend        
        plt.legend(labels = ['Bar Text : obs_dist', 'Dot Text : bad_rate'], loc = 'lower center')
        # Set title
        plt.title(f'Bins Summary Plot - {self.var} \n IV : {(binSummaryTable["iv_grp"].sum()):.4f}')

        # Show the plot
        plt.show()
