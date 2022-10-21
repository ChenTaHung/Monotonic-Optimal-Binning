import pandas as pd
import numpy as np

class mob :
    class monotone :
        def __init__(self, data, var, response, metric) :
            '''
            data : <pandas.DataFrame> 
            var : <string> column to generate metric
            response : <string> column 
            '''
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

        def _mergebins(self, input_df, curtId, nextId) :
            df = input_df.copy()
            newEnd = df.loc[df['id'] == nextId, 'start'].values
            new0 = np.sum(df.loc[df['id'].isin([curtId, nextId]), f"{self.response}_0"].values)
            new1 = np.sum(df.loc[df['id'].isin([curtId, nextId]), f"{self.response}_1"].values)
            newtotal = np.sum([new0, new1])
            newMetric = new1 / newtotal
            df.loc[df['id'] == curtId, ["end", "total", f"{self.response}_0", f"{self.response}_1", self.metric]] = [newEnd, new0, new1, newtotal, newMetric]
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
                                        df, sumDel = self._mergebins(input_df = df, curtId = curtId, nextId = nextId) 
                                        break
                                    else :
                                        sumDel = 0
                                else :
                                    if nextMetric >= curtMetric :
                                        df, sumDel = self._mergebins(input_df = df, curtId = curtId, nextId = nextId)
                                        break
                                    else :
                                        sumDel = 0
                if sumDel == 0 :
                    LoopWhile = False
            
            return df

    class optimalBinning :
        def __init__(self, MonotonicTable, max_bins, min_bins, max_dist, min_dist, max_samples, min_samples, min_bads, init_pvalue, exclude_value = None) :
            self.df = MonotonicTable
            self.max_bins = max_bins
            self.min_bins = min_bins
            self.max_dist = max_dist
            self.min_dist = min_dist
            self.max_samples = max_samples
            self.max_samples = max_samples
            self.min_bads = min_bads
            self.init_pvalue = init_pvalue
            self.exclude_value = exclude_value
            
        def merge(self, mergeTarget : str, mergeType = "MFB") :
            df = self.df.copy()
            if mergeType == "MFB" :
                _isConstraintViolation = True
                
                if df["mergeTarget"].isna().sum() > 0 :
                    _isNaExist = True
                    df_missing = df.iloc[df[mergeTarget].isna(), :]
                else :
                    _isNaExist = False
                
                if self.exclude_value != None :
                    _isExcValueExist = True
                    if isinstance(self.exclude_value, list) :
                        df_xclude_values = df.loc[df[mergeTarget].isin(self.exclude_value), :]
                    elif isinstance(self.exclude_value, (float, int)) :
                        df_xclude_values = df.loc[df[mergeTarget] == self.exclude_value, :]
                else :
                    _isExcValueExist = False
                
                        
                while _isConstraintViolation :
                    
                    _binSummary()
                    
            return woe_summary