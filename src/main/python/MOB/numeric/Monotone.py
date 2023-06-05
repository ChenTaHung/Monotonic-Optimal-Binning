#%%
import pandas as pd
from MOB.numeric.MonotoneNode import MonotoneNode

class Monotone :
    def __init__(self, data, var, response, exclude_value = None, metric = 'mean') :
        '''
        data : <pandas.DataFrame> 
        var : <string> column to generate WoE
        response : <string> column 
        '''
        
        if len(data[response].unique()) != 2:
            raise ValueError(f'More than two unique observations in response Verb : {self.response} ')
        
        if data[response].dtypes == 'object' :
            self.responseType = 'cat'
        else :
            self.responseType = 'num'
        
        self.data = data
        self.var = var
        self.response = response
        self.metric = metric
        self.exclude_value = exclude_value

    def _selectSign(self) -> str:
        '''
        decide `sign` argument
        '''
        if self.data[[self.var, self.response]].corr().iloc[1,0] > 0 :
            sign = '+'
        else :
            sign = '-'
        
        return sign
    
    def _initMonoTable(self) :
        '''
        initialize sorted table to check constraints or reverse trend on metric
        '''
        df = self.data.copy()
        # count = Total
        # sum = Bad
        if df[self.var].isna().sum() == 0 and self.exclude_value == None : # no missing values and no exclude_value specified
            return df.groupby(self.var)[self.response].agg(['count', 'sum', 'std']).reset_index().fillna(0)
        elif df[self.var].isna().sum() != 0 and self.exclude_value == None :
            return df[df[self.var].notnull()].groupby(self.var)[self.response].agg(['count', 'sum', 'std']).reset_index().fillna(0)
        elif df[self.var].isna().sum() == 0 and self.exclude_value != None :
            return df[df[self.var] != self.exclude_value].groupby(self.var)[self.response].agg(['count', 'sum', 'std']).reset_index().fillna(0)
        
    def tuneMonotone(self, initialization : bool = True, sign = 'auto'):
        if sign == 'auto' :
            sign = self._selectSign()
        else :
            sign = sign

        if initialization :
            df = self._initMonoTable()
        else : 
            df = self.data.copy()
            
        root: MonotoneNode = MonotoneNode(0,0,0,0)
        cur: MonotoneNode = root
        
        # assign each node for the double linked list
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
            raise("Invalid sign : <string>  [auto|+|-]")

        # store result
        cur = root
        startValueList = []
        endValueList = []
        binTotalList = []
        binBadList = []
        meanList = []
        stdList = []
        while cur is not None:
            startValueList.append(cur.startValue)
            endValueList.append(cur.endValue)
            binTotalList.append(cur.cumTotal)
            binBadList.append(cur.cumBad)
            stdList.append(cur.cumStd)
            meanList.append(cur.mean)
            cur = cur.next
        
        resDF = pd.DataFrame({'start':startValueList, 'end': endValueList, 'total': binTotalList, 'bads': binBadList, 'mean': meanList, 'std': stdList})
        return resDF
    
# %%
# if __name__ == '__main__' :
#     df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
#     df['default'] = df['default'] - 1
    
#     M = Monotone(data = df, var = 'Durationinmonth', response = 'default')
#     res = M.tuneMonotone()
#     print(res)