import pandas as pd
import numpy as np
from typing import Union
from MOBPY.pav.PAV import PAV
from MOBPY.numeric.OptimalBinning import OptimalBinning

class PAVOB:
    def __init__(self, data, var :str, response : str, exclude_value = None) :
        '''
        Pool Adjacent Violators Algorithm (PAVA)
        
        data : <pandas.DataFrame> 
        var : <string> column for generate the x-axis of the greatest convex minorant (GCM)
        response : <string> column name of the value array created by response_func
        add_var_aggFunc : <dict> Additional columns with corresponding statistics function (similar to pandas.DataFrame.agg())
        exclude_value : <None, list, float, int>
        sign : <str> {+|-|auto} decide the direction (correlation) of var and response variable.
        '''
        self._data = data
        self._var = var
        self._response = response
        # metric : <str> When applying PAVOB, mean is the only valid metric. (default = 'mean') Due to the ttest afterwards in binning process.
        self._metric = 'mean'
        self._exclude_value = exclude_value
        self._OrgDataAssignment = None
        self._CSD_Summary = None
        self._GCM_Summary = None
        self._PAV_Summary = None
        
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
        
        '''
        self._df_sel : data that only contains clean values (no missing values and exclusive values)
        self._df_missing : data that the `var` variable is missing
        self._df_excvalue : data that the `var` variable is exclusive
        '''    
        
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
        
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @property
    def var(self) -> str :
        return self._var
    
    @property
    def response(self) -> str :
        return self._response
    
    @property
    def metric(self) -> str  :
        return self._metric
    
    @property
    def OrgDataAssignment(self) -> Union[pd.DataFrame, None] :
        return self._OrgDataAssignment
    
    @property
    def GCM_Summary(self) -> Union[pd.DataFrame, None] :
        return self._GCM_Summary
    
    @property
    def CSD_Summary(self) -> Union[pd.DataFrame, None] :
        return self._CSD_Summary
    
    @property
    def PAV_Summary(self) -> Union[pd.DataFrame, None] :
        return self._PAV_Summary
    
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
    def isNaExist(self) -> bool :
        return self._isNaExist

    @property
    def isExcValueExist(self) -> bool :
        return self._isExcValueExist
        
    
    def __summarize_GCM_CSD(self, GCM, CSD, MonoTable) :
        '''
        CSD: acceptable inputs : { mean | sum | std | var | min | max | ptp }
            self.var | count | sum | std | max | min
            ----------------------------------------
        ===============================================================
        GCM : 
            intervalStart | intervalEnd | count | sum | std | max | min
            -----------------------------------------------------------
        '''

        # Greatest Convex Minorant Summary -> create metric column.
        _GCM = GCM.copy()
        if self.metric == 'mean' :
            _GCM[self.metric] = _GCM['sum']/_GCM['count']
            
        elif self.metric == 'var' :
            _GCM[self.metric] =  _GCM['std'] ** 2
                   
        elif self.metric == 'ptp' :
            _GCM[self.metric] =  _GCM['max'] - _GCM['min']
        
        else :
            pass
        
        _CSD = CSD.copy()
        _CSD['intervalStart'] = _CSD.apply(lambda row: _GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (_GCM['intervalEnd'] >= row[self.var]), 'intervalStart'].values[0], axis=1)
        _CSD['intervalEnd'] = _CSD.apply(lambda row: _GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (_GCM['intervalEnd'] >= row[self.var]), 'intervalEnd'].values[0], axis=1)
        _CSD['assignMetric'] = _CSD.apply(lambda row: _GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (_GCM['intervalEnd'] >= row[self.var]), self.metric].values[0], axis=1)
      
        return _GCM[['intervalStart', 'intervalEnd', self.metric]], _CSD[[self.var,  f'{self.response}_cum{self.metric.capitalize()}', 'intervalStart', 'intervalEnd', 'assignMetric']]  
    
    def setBinningConstraints(self, max_bins :int = 6, min_bins :int = 4, max_samples = 0.4, min_samples = 0.05, min_pos_term = None, init_pvalue: float = 0.4, maximize_bins :bool = True) -> None:
        self._max_bins = max_bins
        self._min_bins = min_bins
        self._init_pvalue = init_pvalue
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._min_pos_term = min_pos_term
        self._maximize_bins = maximize_bins
        self._constraintsStatus = True
    
    def runPAVOB(self, mergeMethod, sign = 'auto') :
        #Construct PAV object using the `df_sel` data which does not include missing value and exclusive values
        _PAV = PAV(data = self.df_sel, var = self.var, response = self.response, metric = self.metric)
        # PAVA is the Greatest convex minorant of the Cumulative Sum Diagram.
        # initialize the CSD (dataframe) and GCM (dataframe)
        # CSD is the initial groupby dataframe with stats information of response
        # GCM is the result the is tuned as monotonic on metric
        _GCM, _CSD, _MonoTable = _PAV.init_CSD_GCM(sign = sign)

        GCM, CSD = self.__summarize_GCM_CSD(GCM = _GCM, CSD = _CSD, MonoTable = _MonoTable)
        
        OrgDataAssignment = self.df_sel.copy()[[self.var, self.response]]
        OrgDataAssignment['assignValue'] = OrgDataAssignment.apply(lambda row: GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (GCM['intervalEnd'] >= row[self.var]), 'intervalStart'].values[0], axis=1)
        OrgDataAssignment['assignMetric'] = OrgDataAssignment.apply(lambda row: GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (GCM['intervalEnd'] >= row[self.var]), self.metric].values[0], axis=1)

        PAVA_Final_Df = GCM.sort_values(by = 'intervalStart').rename(columns = {'intervalStart':self.var, self.metric : f'{self.response}_{self.metric}'})
        PAVA_Result = PAVA_Final_Df
        #Generate final PAVA result (include additional var aggregation functions)
        # create interval 
        PAVA_Result.insert(1, column='intervalEnd)', value=PAVA_Result[self.var].shift(-1))
        PAVA_Result.rename(columns = {self.var:'[intervalStart'}, inplace=True)
        PAVA_Result.iloc[0, 0] = -np.inf #minimum interval start
        PAVA_Result.iloc[-1, 1] = np.inf #maximum interval end
        PAVA_Result = PAVA_Result.drop('intervalEnd', axis = 1)

        '''
        CSDSummary : a groupby(var) dataset with selected metric column which represent the cumulative sum diagram of the chosen variable and response
        self.var | self.metric
        ----------------------
        '''
        self._CSD_Summary = CSD
        '''
        GCM : the result of the greatest convex minorant diagram that no data violates the convexity.
        intervalStart | intervalEnd | self.metric
        ----------------------------------
        '''
        self._GCM_Summary = GCM
        
        if mergeMethod in ['Stats', 'Size'] :
            if self.constraintsStatus == False :
                raise Exception('Please set the constraints first by using "setBinningConstraints" method.')
            
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
        
