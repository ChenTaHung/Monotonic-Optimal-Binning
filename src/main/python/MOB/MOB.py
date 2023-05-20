import os
os.chdir("/Users/chentahung/Desktop/git/mob-py/src/main/python/MOB/numeric")
from Monotone import Monotone
from OptimalBinning import OptimalBinning

class MOB:
    def __init__(self, data, var, response, target_metric) :
        self.data = data
        self.var = var
        self.response = response 
        self.metric = target_metric
    
    def _setBinningConstraints(self, max_bins :int, min_bins :int, max_samples , min_samples, min_bads, init_pvalue: float) -> None:
        '''
        [attributes]:
        self.max_bins
        self.min_bins
        self.max_samples -> int
        self.min_samples -> int 
        self.min_bads -> int
        '''
        self.max_bins = max_bins
        self.min_bins = min_bins
        self.init_pvalue = init_pvalue
        
        # if given percentage then automatically calculate the maximum number of observation for each bin
        if 0 < max_samples and max_samples <= 1 :
            self.max_samples = self.MonoDf["total"].sum() * max_samples
        else : 
            self.max_samples = max_samples
        # if given percentage then automatically calculate the minimum number of observation for each bin
        if 0 < min_samples and min_samples <= 1 :
            self.min_samples = self.MonoDf["total"].sum() * min_samples
        else : 
            self.min_samples = min_samples
        # if given percentage then automatically calculate the minimum bad number for each bin
        if 0 < min_bads and min_bads <= 1 :
            self.min_bads = self.MonoDf["bad"].sum() * min_bads
        else : 
            self.min_bads = min_bads

    
    def checkTableInfo(self):
        '''
        _isNaExist : check Missing Existance
        _isExcValueExist : check Exclude Value Existance
        
        [attributes]
        self.df_missing : missing data subset
        self.df_excvalue : exclude value data subset
        self.df_sel : selected data subset
        
        '''
        
        # check missing value exist
        if self.data[self.var].isna().sum() > 0 :
            _isNaExist = True
        else :
            _isNaExist = False
        
        # check exclude values exist
        if self.exclude_value != None :
            _isExcValueExist = True
        else :
            _isExcValueExist = False
        
        if _isNaExist & _isExcValueExist :
            self.df_missing = self.data.iloc[self.data[self.var].isna(), :]
            
            if isinstance(self.exclude_value, list) :
                self.df_excvalue = self.data.loc[self.data[self.var].isin(self.exclude_value), :]
            elif isinstance(self.exclude_value, (float, int)) :
                self.df_excvalue = self.data.loc[self.data[self.var] == self.exclude_value, :]
                
            self.df_sel = self.data.loc[(self.data[self.var].notnull()) & (self.data[self.var] != self.exclude_value)]
            
        elif _isNaExist & ~_isExcValueExist: #only contain missing
            self.df_missing = self.data.iloc[self.data[self.var].isna(), :]
            self.df_sel = self.data.loc[self.data[self.var].notnull()]
            
        elif ~_isNaExist & _isExcValueExist : #only contain exclude condition
            if isinstance(self.exclude_value, list) :
                self.df_excvalue = self.data.loc[self.data[self.var].isin(self.exclude_value), :]
            elif isinstance(self.exclude_value, (float, int)) :
                self.df_excvalue = self.data.loc[self.data[self.var] == self.exclude_value, :]
                
            self.df_sel = self.data.loc[self.data[self.var] != self.exclude_value]
        else:
            self.df_sel = self.data
        
        return _isNaExist, _isExcValueExist
            
    
    def runMOB(self) :
        # Check Data Status (Missing and Exclud Value) 
        _isNaExist, _isExcValueExist = checkTableInfo()
        MonotoneTuner = Monotone(data = self.df_sel, var = "Durationinmonth", response = "default")
        MonoTable = MonotoneTuner.tuneMonotone()
        OptimalBinningMerger = OptimalBinning(resMonotoneTable = MonoTable, 
                                              max_bins = self.max_bins, min_bins = self.min_bins, 
                                              max_samples = self.max_samples, min_samples = self.min_samples, 
                                              min_bads = self.min_bads, init_pvalue = self.init_pvalue)
        OptimalBinningMerger.mergeBins()
        if _isNaExist and _isExcValueExist :
            pd.concat([summarizeBin(self.df_missing), summarizeBin(self.df_excvalue), summarizeBin(`d`))])
            
        


            