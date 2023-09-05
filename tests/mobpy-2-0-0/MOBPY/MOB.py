import pandas as pd
import numpy as np
from typing import Union
from MOBPY.numeric.Monotone import Monotone
from MOBPY.numeric.OptimalBinning import OptimalBinning

class MOB:
    def __init__(self, data, var, response, exclude_value = None) :
        self._data = data
        self._var = var
        self._response = response 
        self._exclude_value = exclude_value
        self._constraintsStatus = False
        self._outputTable = None
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
            
        
        if _isNaExist & _isExcValueExist : #both contains missing and exclude values
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            
            if isinstance(exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
                self._df_sel = self._data.loc[(self._data[self._var].notnull()) & ~(self._data[self._var].isin(exclude_value))]
            elif isinstance(exclude_value, (float, int)) :
                self._df_excvalue = self._data.loc[self._data[self._var] == exclude_value, :]
                self._df_sel = self._data.loc[(self._data[self._var].notnull()) & (self._data[self._var] != exclude_value)]
            
        elif _isNaExist & ~_isExcValueExist: #only contain missing
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            self._df_sel = self._data.loc[self._data[self._var].notnull()]
            
        elif ~_isNaExist & _isExcValueExist : #only contain exclude condition
            if isinstance(exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
                self._df_sel = self._data.loc[~self._data[self._var].isin(exclude_value)]
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
    def data(self) -> pd.DataFrame:
        return self._data
    
    @property
    def var(self) -> str :
        return self._var
        
    @property
    def response(self) -> str:
        return self._response
    
    @property
    def exclude_value(self) -> Union[list, int, float, None]:
        return self._exclude_value

    @property
    def constraintsStatus(self) -> bool:
        return self._constraintsStatus
    
    @property
    def isNaExist(self) -> bool:
        return self._isNaExist
    
    @property
    def isExcValueExist(self) -> bool:
        return self._isExcValueExist
    
    @property
    def df_missing(self) -> pd.DataFrame:
        return self._df_missing
    
    @property
    def df_excvalue(self) -> pd.DataFrame:
        return self._df_excvalue
    
    @property
    def df_sel(self) -> pd.DataFrame:
        return self._df_sel
    
    @property
    def max_bins(self) -> int:
        return self._max_bins

    @property
    def min_bins(self) -> int:
        return self._min_bins
        
    @property
    def init_pvalue(self) -> float:
        return self._init_pvalue
        
    @property
    def max_samples(self) -> Union[int, float]:
        return self._max_samples

    @property
    def min_samples(self) -> Union[int, float] :
        return self._min_samples
    
    @property
    def min_bads(self) -> Union[int, float]:
        return self._min_bads
    
    @property
    def maximize_bins(self) -> bool :
        return self._maximize_bins
    
    @property
    def finishBinningTable(self) -> pd.DataFrame :
        return self._finishBinningTable
    
    @property
    def outputTable(self) -> pd.DataFrame :
        return self._outputTable
    
    def setBinningConstraints(self, max_bins :int = 6, min_bins :int = 4, max_samples = 0.4, min_samples = 0.05, min_bads = 0.05, init_pvalue: float = 0.4, maximize_bins :bool = True) -> None:
        self._max_bins = max_bins
        self._min_bins = min_bins
        self._init_pvalue = init_pvalue
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._min_bads = min_bads
        self._maximize_bins = maximize_bins
        self._constraintsStatus = True
           
    def __summarizeBins(self, FinalOptTable) -> pd.DataFrame:
        
        FinalOptTable = FinalOptTable[['[intervalStart', 'total', 'bads', 'mean']].rename(columns = {'total': 'nsamples', 'bad' : 'bads', 'mean' : 'bad_rate'})
        FinalOptTable.insert(1, column = 'intervalEnd)', value = FinalOptTable['[intervalStart'].shift(-1))
        FinalOptTable['dist_obs'] = FinalOptTable['nsamples'] / FinalOptTable['nsamples'].sum()
        FinalOptTable['dist_bads'] = FinalOptTable['bads'] / FinalOptTable['bads'].sum()
        FinalOptTable['goods'] = FinalOptTable['nsamples'] - FinalOptTable['bads']
        FinalOptTable['dist_goods'] = FinalOptTable['goods'] / FinalOptTable['goods'].sum()
        FinalOptTable['woe'] = np.log(FinalOptTable['dist_goods']/FinalOptTable['dist_bads'])
        
        # adjusted woe replacement when encounter zero bads or zero goods
        if (FinalOptTable['bads'] == 0).sum() + (FinalOptTable['goods']).sum() > 0 :
            adj_goods = FinalOptTable.loc[(FinalOptTable['bads'] == 0) | (FinalOptTable['goods'] == 0), 'goods'] + 0.5
            adj_bads = FinalOptTable.loc[(FinalOptTable['bads'] == 0) | (FinalOptTable['goods'] == 0), 'bads'] + 0.5
            adj_dist_goods = adj_goods / FinalOptTable['goods'].sum()
            adj_dist_bads = adj_bads / FinalOptTable['bads'].sum()
            FinalOptTable.loc[(FinalOptTable['bads'] == 0) | (FinalOptTable['goods'] == 0), 'woe'] = np.log(adj_dist_goods/adj_dist_bads)

        FinalOptTable['iv_grp'] = (FinalOptTable['dist_goods'] - FinalOptTable['dist_bads']) * FinalOptTable['woe']
        # continous distribution means the value range from -inf to inf :
        FinalOptTable.iloc[0,0] = -np.inf #first bin interval starts
        FinalOptTable.iloc[-1,1] = np.inf #last bin interval ends
        FinalOptTable.index.name = self.var
        return FinalOptTable      
      
    def runMOB(self, mergeMethod, sign = 'auto') -> pd.DataFrame :
        if mergeMethod in ['Stats', 'Size'] :
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
            finishBinningTable['[intervalStart'] = finishBinningTable['[intervalStart'].astype(str)
            finishBinningTable['intervalEnd)'] = finishBinningTable['intervalEnd)'].astype(str)
            self._finishBinningTable = finishBinningTable.rename(columns = {'intervalEnd)':'intervalEnd]'})
            # Summary
            if self.isNaExist and self.isExcValueExist : #contains missing and exclude value
                
                missingDF = pd.DataFrame({
                    '[intervalStart' : ['Missing'],
                    'intervalEnd)' : ['Missing'],
                    'total' : [len(self.df_missing)],
                    'bads' : [self.df_missing[self.response].sum()],
                    'mean' : [(self.df_missing[self.response].sum()) / (len(self.df_missing))]})
                    
                excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename(columns={self.var: '[intervalStart','count':'total', 'sum':'bads'})
                excludeValueDF['[intervalStart'] = excludeValueDF['[intervalStart'].astype(str)
                excludeValueDF.insert(1, 'intervalEnd)', excludeValueDF['[intervalStart'])
                excludeValueDF['mean'] = excludeValueDF['bads'] / excludeValueDF['total']
                
                completeBinningTable = pd.concat([finishBinningTable, missingDF, excludeValueDF], axis = 0, ignore_index = True)
            elif self.isNaExist & ~self.isExcValueExist : # contains missing but no special values
                missingDF = pd.DataFrame({
                    '[intervalStart' : ['Missing'],
                    'intervalEnd)' : ['Missing'],
                    'total' : [len(self.df_missing)],
                    'bads' : [self.df_missing[self.response].sum()],
                    'mean' : [(self.df_missing[self.response].sum()) / (len(self.df_missing))]})
                
                completeBinningTable = pd.concat([finishBinningTable, missingDF], axis = 0, ignore_index = True)
            elif ~self.isNaExist & self.isExcValueExist : # contains special values but no missing data
                excludeValueDF = self.df_excvalue.groupby(self.var)[self.response].agg(['count', 'sum']).reset_index().fillna(0).rename(columns={self.var: '[intervalStart','count':'total', 'sum':'bads'})
                excludeValueDF['[intervalStart'] = excludeValueDF['[intervalStart'].astype(str)
                excludeValueDF.insert(1, 'intervalEnd', excludeValueDF['[intervalStart'])
                excludeValueDF['mean'] = excludeValueDF['bads'] / excludeValueDF['total']
                
                completeBinningTable = pd.concat([finishBinningTable, excludeValueDF], axis = 0, ignore_index = True)
            else : # clean data with no missing and special values
                completeBinningTable = finishBinningTable
                
            outputTable = self.__summarizeBins(FinalOptTable = completeBinningTable)
            # intervalStart and intervalEnd is set as string due to the concatenation of missing and exclusive dataset
            self._outputTable = outputTable
        else :
            raise ValueError('mergeMethod only supports {<Stats> | <Size>} so far.')
                
        return outputTable

    def applyMOB(self, var_column : pd.Series, assign :str = 'interval') :
        '''
        assignment: <str> {'interval'|'start'|'end'}
        '''
        _MOBout = self.outputTable
        _MOBout['[intervalStart'] = _MOBout['[intervalStart'].astype(float)
        _MOBout['intervalEnd)'] = _MOBout['intervalEnd)'].astype(float)
        if assign == 'interval' :
            # include intervalStart but exclude intervalEnd :
            MOB_Res_Series = var_column.apply(lambda row : f'[ {_MOBout.loc[(_MOBout["[intervalStart"] <= row)&(_MOBout["intervalEnd)"] > row), "[intervalStart"].values[0]} , {_MOBout.loc[(_MOBout["[intervalStart"] <= row)&(_MOBout["intervalEnd)"] > row), "intervalEnd)"].values[0]} )')
        elif assign == 'start' :
            MOB_Res_Series = var_column.apply(lambda row : _MOBout.loc[(_MOBout["[intervalStart"] <= row)&(_MOBout["intervalEnd)"] > row), "[intervalStart"].values[0])
        elif assign == 'end' :
            MOB_Res_Series = var_column.apply(lambda row : _MOBout.loc[(_MOBout["[intervalStart"] <= row)&(_MOBout["intervalEnd)"] > row), "intervalEnd)"].values[0])
        
        return MOB_Res_Series