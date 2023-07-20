
from typing import Union, Callable
import pandas as pd
import numpy as np
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src')
from MOBPY.pav.PavMonoNode import PavMonoNode

class PAV :
    def __init__(self, data, var :str, response : str, metric : str) :
        '''
        PAVa : Pool Adjacent Violators algorithm
        
        data : <pandas.DataFrame> 
        var : <string> column for generate the x-axis of the greatest convex minorant (GCM)
        response : <string> column name of the value array created by response_func
        metric : <callable> The Statistics metric of the response variable
        '''
        self._data = data
        self._var = var
        self._response = response
        self._metric = metric
        # self._CSD = None
        # self._GCM = None
        
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
  
    # @property
    # def CSD(self) -> pd.DataFrame :
    #     return self._CSD

    # @property
    # def GCM(self) -> pd.DataFrame :
    #     return self._GCM

    def __selectSign(self) -> str:
        '''
        decide `sign` argument
        '''
        if self.data[[self.var, self.response]].corr().iloc[1,0] > 0 :
            sign = '+'
        else :
            sign = '-'
        
        return sign
    
    def __initMonoTable(self) -> pd.DataFrame:
        '''
        initialize Cumulative Sum Diagram (CSD)
        '''
        df = self.data.copy()
        MonoTable = df.groupby(self.var)[self.response].agg(['count', 'sum', 'std', 'max', 'min']).reset_index().fillna(0)

        return MonoTable
    
    def __initGCM(self, MonoTable, sign = 'auto') -> pd.DataFrame:
        '''
        create greatest convex minorant (GCM) of the Cumulative Sum Diagram (CSD)
        
        Relevant
        '''
        
        # TODO
        root: PavMonoNode = PavMonoNode(value = 0, stats_array = np.array([0, 0, 0, 0, 0]), metric = self.metric)
        cur: PavMonoNode = root
        
        # assign each node for the double linked list
        # stats_array = ['count', 'sum', 'std', 'max', 'min']
        for row in MonoTable.values:
            _tmp = PavMonoNode(value = row[0], stats_array = np.array([row[1], row[2], row[3], row[4], row[5]]), metric = self.metric) #define every node 
            cur.next = _tmp
            _tmp.pre = cur
            cur = cur.next

        CSD = []

        root = root.next # start from index = 1, `cur` compares with `cur.pre`
        root.pre = None

        # start comparison and merging
        cur: PavMonoNode = root
        cur = cur.next
        
        if sign == '+' :
            # if cur.pre is not None :
            #     CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                
            while cur is not None : #not the last node
                while cur.pre is not None and cur.stats <= cur.pre.stats: # not the first one and strictedly and ascendingly monotonic
                    cur.pre.update(nextValue = cur.endValue, nextCount = cur.count, nextSum = cur.sum, nextStd = cur.std, nextMax = cur.max, nextMin = cur.min) # TODO
                    CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                    cur.pre.next = cur.next
                    if cur.next is not None:
                        cur.next.pre = cur.pre
                    cur = cur.pre
                cur = cur.next
        elif sign == '-' :
            # if cur.pre is not None :
            #     CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                
            while cur is not None : #not the last node
                while cur.pre is not None and cur.stats >= cur.pre.stats: # not the first one and strictedly and descendingly monotonic
                    cur.pre.update(nextValue = cur.endValue, nextCount = cur.count, nextSum = cur.sum, nextStd = cur.std, nextMax = cur.max, nextMin = cur.min) # TODO
                    CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                    cur.pre.next = cur.next
                    if cur.next is not None:
                        cur.next.pre = cur.pre
                    cur = cur.pre
                cur = cur.next
        else :
            raise("Invalid sign : <string>  {auto,+,-}")

        # store result
        cur = root
        startList = []
        endList = []
        countList = []
        sumList = []
        stdList = []
        maxList = []
        minList = []
        while cur is not None:
            startList.append(cur.startValue)
            endList.append(cur.endValue)
            countList.append(cur.count)
            sumList.append(cur.sum)
            stdList.append(cur.std)
            maxList.append(cur.max)
            minList.append(cur.min)
            cur = cur.next
        
        resDF = pd.DataFrame({'intervalStart':startList, 'intervalEnd': endList, 'count': countList, 'sum': sumList,'std': stdList, 'max':maxList, 'min' : minList})
        CSD = pd.DataFrame(CSD, columns =[self.var, 'window_cumCount', 'window_cumSum', 'window_cumStd', 'window_cumMax', 'window_cumMin', f'window_cum{self.metric.capitalize()}'])
        return resDF, CSD

    def init_CSD_GCM(self, sign = 'auto') :
        
        if sign == 'auto' :
            sign = self.__selectSign()
        else :
            sign = sign

        # initialize Cumulative Sum Diagram (CSD)
        MonoTable = self.__initMonoTable()   
        # initialize Greatest Convex Minorant (GCM)
        GCM, CSD = self.__initGCM(MonoTable = MonoTable, sign = sign)
        
        return GCM, CSD
        
        
    # def __missing_exclusive_summary(self) -> None:
    #     if self.isNaExist and self.isExcValueExist : #missing and exclusive
    #         miss_exc_agg_dict = self.add_var_aggFunc.copy()
    #         miss_exc_agg_dict[self.response] = self.metric
    #         '''
    #         Missing Data Section
    #         '''
    #         _df_missing = self.df_missing.copy()

    #         # CSD2GCMSummary
    #         _dfm_CSD2GCM = pd.DataFrame({'self.var' : [np.nan], 
    #                                      self.response : _df_missing[self.response].agg(self.metric).to_list(), 
    #                                      'assignValue' :[np.nan], 
    #                                      'assignMetric' : _df_missing[self.response].agg(self.metric).to_list()})
            
    #         # CSDSummary
    #         _dfm_CSD = _df_missing.groupby(self.var, dropna = False).agg(self.metric).reset_index()
    #         _dfm_CSD.columns = [self.var, self.metric]
            
    #         # Pava Summary
    #         # create aggregate dictionary
    #         _dfm_PAVA = _df_missing.groupby(self.var, dropna = False).agg(miss_exc_agg_dict)
    #         _dfm_PAVA.columns = newColName
    #         '''
    #         Exclusive Data Section
    #         '''
    #         _df_exc = self.df_excvalue.copy()
            
    #         # CSD Summary
    #         _dfe_CSD = _df_exc.groupby(self.var)[self.response].agg(self.metric).reset_index()
    #         _dfe_CSD.columns = [self.var, self.metric]
            
    #         # CSD2GCM 
    #         mapping_dict = dict(zip(_dfe_CSD[self.var], _dfe_CSD[self.metric]))
    #         _dfe_CSD2GCM = _df_exc[[self.var, self.response]]
    #         _dfe_CSD2GCM['assignValue'] = _dfe_CSD2GCM[self.var]
    #         _dfe_CSD2GCM['assignValue'] = _dfe_CSD2GCM[self.var].map(mapping_dict)
            
    #         # PAVA Summary
    #         _dfe_PAVA = _df_exc.groupby(self.var)[self.response].agg(miss_exc_agg_dict).reset_index()
    #         _dfe_PAVA.columns = newColName
            
    #     elif self.isNaExist and ~self.isExcValueExist : #only contains missing
    #         miss_exc_agg_dict = self.add_var_aggFunc.copy()
    #         miss_exc_agg_dict[self.response] = self.metric
    #         '''
    #         Missing Data Section
    #         '''
    #         _df_missing = self.df_missing.copy()

    #         # CSD2GCMSummary
    #         _dfm_CSD2GCM = pd.DataFrame({'self.var' : [np.nan], 
    #                                      self.response : _df_missing[self.response].agg(self.metric).to_list(), 
    #                                      'assignValue' :[np.nan], 
    #                                      'assignMetric' : _df_missing[self.response].agg(self.metric).to_list()})
            
    #         # CSDSummary
    #         _dfm_CSD = _df_missing.groupby(self.var, dropna = False).agg(self.metric).reset_index()
    #         _dfm_CSD.columns = [self.var, self.metric]
            
    #         # Pava Summary
    #         # create aggregate dictionary
    #         _dfm_PAVA = _df_missing.groupby(self.var, dropna = False).agg(miss_exc_agg_dict)
    #         _dfm_PAVA.columns = newColName
        
    #     elif ~self.isNaExist and self.isExcValueExist : #only contains exclusive values
    #         '''
    #         Exclusive Data Section
    #         '''
    #         _df_exc = self.df_excvalue.copy()
            
    #         # CSD Summary
    #         _dfe_CSD = _df_exc.groupby(self.var)[self.response].agg(self.metric).reset_index()
    #         _dfe_CSD.columns = [self.var, self.metric]
            
    #         # CSD2GCM 
    #         mapping_dict = dict(zip(_dfe_CSD[self.var], _dfe_CSD[self.metric]))
    #         _dfe_CSD2GCM = _df_exc[[self.var, self.response]]
    #         _dfe_CSD2GCM['assignValue'] = _dfe_CSD2GCM[self.var]
    #         _dfe_CSD2GCM['assignValue'] = _dfe_CSD2GCM[self.var].map(mapping_dict)
            
    #         # PAVA Summary
    #         _dfe_PAVA = _df_exc.groupby(self.var)[self.response].agg(miss_exc_agg_dict).reset_index()
    #         _dfe_PAVA.columns = newColName
            
    #     else :
    #         pass
