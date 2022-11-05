import pandas as pd
import numpy as np
from mobFunc import MobFunc

class Monotone(MobFunc) :
    def __init__(self, data, var, response, metric) :
        '''
        data : <pandas.DataFrame> 
        var : <string> column to generate metric
        response : <string> column 
        '''
        super().__init__()
        
        if len(data[response].unique()) != 2:
            raise ValueError(f"More than two unique observations in response Verb : {self.response} ")

        if data[var].dtypes == "object" :
            raise ValueError(f"Variable : {var} is not a numerical variable")
        
        if data[response].dtypes == "object" :
            self.responseType = "cat"
        else :
            self.responseType = "num"
        
        self.data = data
        self.var = var
        self.response = response
        self.metric = metric

    def _selectSign(self) :
        if self.data[[self.var, self.response]].corr().iloc[1,0] > 0 :
            sign = "+"
        else :
            sign = "-"
        
        return sign
    
    def _initMonoTable(self) :
        df = self.data.copy()
        
        df1 = df.groupby(self.var)[self.response].agg(["count", "sum", self.metric]).sort_index().reset_index()
        df1.insert(1, "end", df1[self.var])
        
        df_result = df1.rename(columns = {self.var : "start", "count" : "total", "sum" : f"{self.response}_1"})
        df_result[f"{self.response}_0"] = df_result["total"] - df_result[f"{self.response}_1"]
        df_result.index.name = self.var
        
        return df_result

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

    def tuneMonotone(self, monoTarget, initialization : bool = True, sign = "auto") :
        if sign == "auto" :
            sign = self._selectSign()
        else :
            sign = sign

        if initialization :
            df = self._initMonoTable()
        else : 
            df = self.data.copy()
            
        df['del'] = np.zeros(len(df))
        df['id'] = [*range(1, len(df)+1)]
        
        LoopWhile = True
        
        while LoopWhile :
            idList = df['id'].to_list()

            if len(idList) == 1 : # only one bin left
                LoopWhile = False
            else :
                for i in range(len(idList) + 1) :
                    if i == len(idList) - 1 : # current row is the last row
                        break
                    else :
                        curtId = idList[i]
                        nextId = idList[i + 1]
                        if df.loc[df['id'] == curtId, 'del'].values == 1:
                            continue
                        else :
                            curtMetric = df.loc[df['id'] == curtId, monoTarget].values
                            nextMetric = df.loc[df['id'] == nextId, monoTarget].values
                            if sign == "+" :
                                if nextMetric <= curtMetric :
                                    df, sumDel = self._mergeMonoBins(input_df = df, curtId = curtId, nextId = nextId) 
                                    break
                                else :
                                    sumDel = 0
                            else :
                                if nextMetric >= curtMetric :
                                    df, sumDel = self._mergeMonoBins(input_df = df, curtId = curtId, nextId = nextId)
                                    break
                                else :
                                    sumDel = 0
            if sumDel == 0 :
                LoopWhile = False
        
        return df
    
    
''' 
Low correlation relation between x and y will cost heavy computation on no matter monotoning or binning.
Due to the number of unique values and the corr :
    1. Initially bucket the variable if the variable contains too many unique values
    2. Filter by corr
    
Pseudo Code (optimize speed)
        
def setDelete(curMean, nexMean, curID = 1, nexId = 1, delFlag = 1) :
    if nexMean < curMean : // not monotonic set the del flag
        CurMean = mean(CurMean, NexMean)
        nexID = nexID + 1
        setDelete(curMean, nexMean, curID = curID, nexId = nexId, delFlag = delFlag)
    else : // follow monotonic trend, jump to the next bin as the current bin
        curMean = NexMean
        curID = nexID
        delFlag = delFlag + 1
        setDelete(curMean, nexMean, curID = curID, nexId = nexId, delFlag = delFlag)
    
    return void
        
if __name__ == "__main"" :
    i = index[0]
    curMean = mean[i]
    nexMean = mean[i+1]
    setDelete(curMean, nexMean)
    
    if delFlag 一樣 : merge bins with same delFlag to their previous bin 
            
            
'''