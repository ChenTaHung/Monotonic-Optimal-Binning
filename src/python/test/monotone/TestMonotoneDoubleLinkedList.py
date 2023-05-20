import pandas as pd
import numpy as np
import os
os.chdir("/Users/chentahung/Desktop/git/mob-py/src/python/test/monotone")
from Node import Node

class Monotone :
    def __init__(self, data, var, response, metric = "mean") :
        '''
        data : <pandas.DataFrame> 
        var : <string> column to generate WoE
        response : <string> column 
        '''
        
        if len(data[response].unique()) != 2:
            raise ValueError(f"More than two unique observations in response Verb : {self.response} ")
        
        if data[response].dtypes == "object" :
            self.responseType = "cat"
        else :
            self.responseType = "num"
        
        self.data = data
        self.var = var
        self.response = response
        self.metric = metric

    def _selectSign(self) :
        '''
        decide `sign` argument
        '''
        if self.data[[self.var, self.response]].corr().iloc[1,0] > 0 :
            sign = "+"
        else :
            sign = "-"
        
        return sign
    
    def _initMonoTable(self) :
        '''
        initialize sorted table to check constraints or reverse trend on metric
        '''
        df = self.data.copy()
        
        return df.groupby(self.var)[self.response].agg(["count", "sum"]).reset_index().rename(columns = {"count":"Total", "sum" :"Bad"})

    def _mergeMonoBins(self, input_df, curtId, nextId) :
        df = input_df.copy()
        newEnd = df.loc[df['id'] == nextId, 'end'].values
        new0 = np.sum(df.loc[df['id'].isin([curtId, nextId]), f"{self.response}_0"].values)
        new1 = np.sum(df.loc[df['id'].isin([curtId, nextId]), f"{self.response}_1"].values)
        newtotal = np.sum([new0, new1])
        newMetric = new1 / newtotal
        df.loc[df['id'] == curtId, ["end", "total", f"{self.response}_0", f"{self.response}_1", self.metric]] = np.array([newEnd, newtotal, new0, new1, newMetric], dtype=object)
        df.loc[df['id'] == nextId, "del"] = 1
        sumDel = df["del"].sum()
        df = df.drop(df[df['del'] == 1].index)
        
        return df, sumDel

    def tuneMonotone(self, initialization : bool = True, sign = "auto") :
        if sign == "auto" :
            sign = self._selectSign()
        else :
            sign = sign

        if initialization :
            df = self._initMonoTable()
        else : 
            df = self.data.copy()
            
        root: Node = Node(0, 0, 0)
        cur: Node = root
        
        for i, row in enumerate(df.values):
            _tmp = Node(row[0], row[1], row[2],) #define every node
            cur.next = _tmp
            _tmp.pre = cur
            cur = cur.next
        
        root = root.next # start from index = 1, `cur` compares with `cur.pre`
        root.pre = None

        # start comparison and merging
        cur: Node = root
        cur = cur.next
        while cur is not None :
            while cur.pre is not None and cur.mean <= cur.pre.mean:
                cur.pre.update(cur.mergeTotal, cur.mergeBad, cur.endValue)
                cur.pre.next = cur.next
                if cur.next is not None:
                    cur.next.pre = cur.pre
                cur = cur.pre
            cur = cur.next

        # print result
        cur = root
        startValueList = []
        endValueList = []
        binTotalList = []
        binBadList = []
        meanList = []
        while cur is not None:
            startValueList.append(cur.startValue)
            endValueList.append(cur.endValue)
            binTotalList.append(cur.binTotal)
            binBadList.append(cur.binBad)
            meanList.append(cur.mean)
            cur = cur.next
            
        return pd.DataFrame({"start":startValueList, "end": endValueList, "total": binTotalList, "bad": binBadList, "mean": meanList})
                
if __name__ == "__main__" :
    df = pd.read_csv("/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv")
    M = Monotone(df, var = "Durationinmonth", response = "default")
    M.tuneMonotone()