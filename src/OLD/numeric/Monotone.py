#%%
import pandas as pd
from typing import Union
import os
# os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python')
from MOBPY.numeric.MonotoneNode import MonotoneNode

class Monotone :
    def __init__(self, data, var, response, metric = 'mean') :
        '''
        data : <pandas.DataFrame> 
        var : <string> column to generate WoE
        response : <string> column 
        '''
        
        if len(data[response].unique()) != 2:
            raise ValueError(f'More than two unique observations in response Verb : {response}')
        
        if data[response].dtypes == 'object' :
            raise ValueError(f'Please change the response variable into numeric with `1` represent positive and `0` represent the other')
        
        self._data = data
        self._var = var
        self._response = response
        self._metric = metric
            

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
    def metric(self) -> str :
        return self._metric
    
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
        initialize sorted table to check constraints or reverse trend on metric
        '''
        df = self.data.copy()

        return df.groupby(self.var)[self.response].agg(['count', 'sum', 'std']).reset_index().fillna(0)
    
    def tuneMonotone(self, initialization : bool = True, sign = 'auto') -> pd.DataFrame:
        if sign == 'auto' :
            sign = self.__selectSign()
        else :
            sign = sign

        if initialization :
            df = self.__initMonoTable()
        else : 
            df = self.data.copy()
            
        root: MonotoneNode = MonotoneNode(0,0,0,0)
        cur: MonotoneNode = root
        
        # assign each node for the double linked list
  
        # TODO : take `MonotoneNode` out of the for loop to new the object only once :
        # TODO :    we update the element in the MonotoneNode in the for loop. 
        # TODO :    Delete the value of the variable after every update.    
     
        for row in df.values:
            _tmp = MonotoneNode(Value = row[0], FirstTotal = row[1], FirstBad = row[2], FirstStd = row[3]) #define every node
            cur.next = _tmp
            _tmp.pre = cur
            cur = cur.next


        root = root.next # start from index = 1, `cur` compares with `cur.pre`
        root.pre = None

        # start comparison and merging
        cur: MonotoneNode = root
        cur = cur.next
        
        if sign == '+' :
            while cur is not None : #not the last node
                while cur.pre is not None and cur.mean <= cur.pre.mean: # not the first one and strictedly and ascendingly monotonic
                    cur.pre.update(cur.mergeTotal, cur.mergeBad, cur.mergeStd, cur.endValue)
                    cur.pre.next = cur.next
                    if cur.next is not None:
                        cur.next.pre = cur.pre
                    cur = cur.pre
                cur = cur.next
        elif sign == '-' :
            while cur is not None : #not the last node
                while cur.pre is not None and cur.mean >= cur.pre.mean: # not the first one and strictedly and descendingly monotonic
                    cur.pre.update(cur.mergeTotal, cur.mergeBad, cur.mergeStd, cur.endValue)
                    cur.pre.next = cur.next
                    if cur.next is not None:
                        cur.next.pre = cur.pre
                    cur = cur.pre
                cur = cur.next
        else :
            raise("Invalid sign : <string>  {auto,+,-}")

        # store result
        cur = root
        startValueList = []
        endValueList = []
        binTotalList = []
        binBadList = []
        meanList = []
        stdList = []
        while cur is not None:
            startValueList.append(str(cur.startValue))
            endValueList.append(str(cur.endValue))
            binTotalList.append(cur.cumTotal)
            binBadList.append(cur.cumBad)
            stdList.append(cur.cumStd)
            meanList.append(cur.mean)
            cur = cur.next
        
        resDF = pd.DataFrame({'[intervalStart':startValueList, 'intervalEnd)': endValueList, 'total': binTotalList, 'bads': binBadList, 'mean': meanList, 'std': stdList})
        return resDF
    
# %%
# if __name__ == '__main__' :
#     df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
#     df['default'] = df['default'] - 1
    
#     M = Monotone(data = df, var = 'Durationinmonth', response = 'default')
#     res = M.tuneMonotone()
#     print(res)
# %%
