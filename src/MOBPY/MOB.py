import pandas as pd
import numpy as np
from MOBPY.numeric.Monotone import Monotone
from MOBPY.numeric.OptimalBinning import OptimalBinning

class MOB:
    def __init__(self, data, var, response, exclude_value = None) :
        self._data = data
        self._var = var
        self._response = response 
        self._exclude_value = exclude_value
        self._constraintsStatus = False
        '''
        _isNaExist : check Missing Existance
        _isExcValueExist : check Exclude Value Existance
        
        [attributes]
        self.df_missing : missing data subset
        self.df_excvalue : exclude value data subset
        self.df_sel : selected data subset
        
        '''
        if self._data[self._var].isna().sum() > 0 :
            _isNaExist = True
        else :
            _isNaExist = False
        
        # check exclude values exist
        if isinstance(exclude_value, list) :
            if self._data[self._var].isin(exclude_value).sum() > 0 : #contains exclude value
                _isExcValueExist = True
            else :
                _isExcValueExist = False
        elif isinstance(exclude_value, (float, int)) :
            if self._data[self._var].isin([exclude_value]).sum() > 0 :
                _isExcValueExist = True
            else :
                _isExcValueExist = False
        elif exclude_value == None :
            _isExcValueExist = False
        else :
            _isExcValueExist = False
            
        
        if _isNaExist & _isExcValueExist :
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            
            if isinstance(self._exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
            elif isinstance(exclude_value, (float, int)) :
                self._df_excvalue = self._data.loc[self._data[self._var] == exclude_value, :]
                
            self._df_sel = self._data.loc[(self._data[self._var].notnull()) & (self._data[self._var] != exclude_value)]
            
        elif _isNaExist & ~_isExcValueExist: #only contain missing
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            self._df_sel = self._data.loc[self._data[self._var].notnull()]
            
        elif ~_isNaExist & _isExcValueExist : #only contain exclude condition
            if isinstance(exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
            elif isinstance(exclude_value, (float, int)) :
                self._df_excvalue = self._data.loc[self._data[self._var] == exclude_value, :]
                
            self._df_sel = self._data.loc[self._data[self._var] != exclude_value]
        else:
            self._df_sel = self._data
            
        self._isNaExist = _isNaExist
        self._isExcValueExist = _isExcValueExist
        # binning constraints
        self._max_bins = 6
        self._min_bins = 6
        self._init_pvalue = 0.4
        self._max_samples = 0.4
        self._min_samples = 0.05
        self._min_bads = 0.05
        self._maximize_bins = True
        self._finishBinningTable = None
        
    @property
    def data(self) :
        return self._data
    @data.setter
    def data(self, value) :
        self._data = value
    
    @property
    def var(self):
        return self._var
        
    @property
    def response(self):
        return self._response
    
    @property
    def exclude_value(self):
        return self._exclude_value

    @property
    def constraintsStatus(self):
        return self._constraintsStatus
    
    @property
    def isNaExist(self):
        return self._isNaExist
    
    @property
    def isExcValueExist(self) :
        return self._isExcValueExist
    
    @property
    def df_missing(self):
        return self._df_missing
    
    @property
    def df_excvalue(self):
        return self._df_excvalue
    
    @property
    def df_sel(self):
        return self._df_sel
    
    @property
    def max_bins(self) :
        return self._max_bins
    # @max_bins.setter
    # def max_bins(self, value):
    #     self._max_bins = value

    @property
    def min_bins(self):
        return self._min_bins
        
    @property
    def init_pvalue(self) :
        return self._init_pvalue
        
    @property
    def max_samples(self) :
        return self._max_samples

    @property
    def min_samples(self) :
        return self._min_samples
    
    @property
    def min_bads(self):
        return self._min_bads
    
    @property
    def maximize_bins(self) :
        return self._maximize_bins
    
    @property
    def finishBinningTable(self) :
        return self._finishBinningTable
    
    def setBinningConstraints(self, max_bins :int = 6, min_bins :int = 4, max_samples = 0.4, min_samples = 0.05, min_bads = 0.05, init_pvalue: float = 0.4, maximize_bins :bool = True) -> None:
        self._max_bins = max_bins
        self._min_bins = min_bins
        self._init_pvalue = init_pvalue
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._min_bads = min_bads
        self._maximize_bins = maximize_bins
        self._constraintsStatus = True
           
    def __summarizeBins(self, FinalOptTable):
        
        FinalOptTable = FinalOptTable[['start', 'end', 'total', 'bads', 'mean']].rename(columns = {'total': 'nsamples', 'bad' : 'bads', 'mean' : 'bad_rate'})
        FinalOptTable['dist_obs'] = FinalOptTable['nsamples'] / FinalOptTable['nsamples'].sum()
        FinalOptTable['dist_bads'] = FinalOptTable['bads'] / FinalOptTable['bads'].sum()
        FinalOptTable['goods'] = FinalOptTable['nsamples'] - FinalOptTable['bads']
        FinalOptTable['dist_goods'] = FinalOptTable['goods'] / FinalOptTable['goods'].sum()
        FinalOptTable['woe'] = np.log(FinalOptTable['dist_goods']/FinalOptTable['dist_bads'])
        FinalOptTable['iv_grp'] = (FinalOptTable['dist_goods'] - FinalOptTable['dist_bads']) * FinalOptTable['woe']
        
        # TODO : Missing Data bads = 0 時候 woe 計算 (exc_value 同理） adjusted woe
        return FinalOptTable      
      
    def runMOB(self, mergeMethod, sign = 'auto') :
        if self.constraintsStatus == False :
            raise Exception('Please set the constraints first by using "setBinningConstraints" method.')
        
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
        finishBinningTable['start'] = finishBinningTable['start'].astype(str)
        finishBinningTable['end'] = finishBinningTable['end'].astype(str)
        self._finishBinningTable = finishBinningTable
        # Summary
        if self.isNaExist and self.isExcValueExist : #contains missing and exclude value
            
            missingDF = pd.DataFrame({
                'start' : ['Missing'],
                'end' : ['Missing'],
                'total' : [len(self.df_missing)],
                'bads' : [self.df_missing[self.response].sum()],
                'mean' : [(self.df_missing[self.response].sum()) / (len(self.df_missing))]})
                
            excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename({self.var: 'start','count':'total', 'sum':'bads'})
            excludeValueDF['start'] = excludeValueDF['start'].astype(str)
            excludeValueDF.insert(1, 'end', excludeValueDF['start'])
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
            excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename({self.var: 'start','count':'total', 'sum':'bads'})
            excludeValueDF['start'] = excludeValueDF['start'].astype(str)
            excludeValueDF.insert(1, 'end', excludeValueDF['start'])
            excludeValueDF['mean'] = excludeValueDF['bads'] / excludeValueDF['total']
            
            completeBinningTable = pd.concat([finishBinningTable, excludeValueDF], axis = 0, ignore_index = True)
        else : # clean data with no missing and special values
            completeBinningTable = finishBinningTable
            
        outputTable = self.__summarizeBins(FinalOptTable = completeBinningTable)
        
        return outputTable
