from typing import Union, Callable
import pandas as pd
import numpy as np
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

    def __selectPAVASign(self, MonoTable) -> str:
        '''
        decide `sign` argument
        '''
        if self.metric == 'count' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['count'])
        
        elif self.metric == 'mean' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['sum'] / MonoTable['count'])
        
        elif self.metric == 'sum' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['sum'])
        
        elif self.metric == 'std' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['std'])
        
        elif self.metric == 'var' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['std']**2)
        
        elif self.metric == 'min' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['min'])
        
        elif self.metric == 'max' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['max'])
        
        elif self.metric == 'ptp' :
            pearson_corr = MonoTable[self.var].corr(MonoTable['max']-MonoTable['min'])
        
        else :
            raise ValueError(f'Invalid metric: {self.metric}, choose one of the following values:\n {{ mean | sum | std | var | min | max | ptp }}')
        
        if pearson_corr > 0 :
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


        root = root.next # start from index = 1, `cur` compares with `cur.pre`
        root.pre = None

        # start comparison and merging
        cur: PavMonoNode = root
        cur = cur.next

        CSD = [tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats])]
        if sign == '+' :
            # if cur.pre is not None :
            #     CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                
            while cur is not None : #not the last node
                while cur.pre is not None :
                    if cur.stats <= cur.pre.stats: # not the first one and strictedly and ascendingly monotonic
                        cur.pre.update(nextValue = cur.endValue, nextCount = cur.count, nextSum = cur.sum, nextStd = cur.std, nextMax = cur.max, nextMin = cur.min) # TODO
                        CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                        cur.pre.next = cur.next
                        if cur.next is not None:
                            cur.next.pre = cur.pre
                        cur = cur.pre
                    else :
                        CSD.append(tuple([cur.startValue] + list(cur.stats_array) + [cur.stats]))
                        break
                    
                cur = cur.next
                
        elif sign == '-' :
            # if cur.pre is not None :
            #     CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                
            while cur is not None : #not the last node
                while cur.pre is not None :
                    if cur.stats >= cur.pre.stats: # not the first one and strictedly and descendingly monotonic
                        cur.pre.update(nextValue = cur.endValue, nextCount = cur.count, nextSum = cur.sum, nextStd = cur.std, nextMax = cur.max, nextMin = cur.min) # TODO
                        CSD.append(tuple([cur.pre.endValue] + list(cur.pre.stats_array) + [cur.pre.stats]))
                        
                        cur.pre.next = cur.next
                        if cur.next is not None:
                            cur.next.pre = cur.pre
                        cur = cur.pre
                    else :
                        CSD.append(tuple([cur.startValue] + list(cur.stats_array) + [cur.stats]))
                        break
                    
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
        csdDF = pd.DataFrame(CSD, columns =[self.var, f'{self.response}_cumCount',  f'{self.response}_cumSum',  f'{self.response}_cumStd',  f'{self.response}_cumMax',  f'{self.response}_cumMin',  f'{self.response}_cum{self.metric.capitalize()}'])
        if self.metric in ['count', 'sum', 'std', 'max', 'min']:
            csdDF = csdDF.loc[:, ~csdDF.columns.duplicated()].copy()
        
        return resDF, csdDF

    def init_CSD_GCM(self, sign = 'auto') -> Union[tuple, pd.DataFrame]:
        
        # initialize Cumulative Sum Diagram (CSD)
        MonoTable = self.__initMonoTable()   
        
        if sign == 'auto' :
            sign = self.__selectPAVASign(MonoTable = MonoTable)
        else :
            sign = sign

        # initialize Greatest Convex Minorant (GCM)
        GCM, CSD = self.__initGCM(MonoTable = MonoTable, sign = sign)
        
        return GCM, CSD, MonoTable

