#%%
import pandas as pd
import numpy as np
import scipy.stats

class OptimalBinning :
    def __init__(self, resMonotoneTable, max_bins :int, min_bins:int, max_samples, min_samples, min_bads, init_pvalue :float, maximize_bins :bool) :
        self._MonoDf = resMonotoneTable
        self._max_bins = max_bins
        self._min_bins = min_bins
        # if given percentage then automatically calculate the maximum number of observation for each bin
        if 0 < max_samples and max_samples <= 1 :
            self._max_samples = self._MonoDf['total'].sum() * max_samples
        else : 
            self._max_samples = max_samples
        # if given percentage then automatically calculate the minimum number of observation for each bin
        if 0 < min_samples and min_samples <= 1 :
            self._min_samples = self._MonoDf['total'].sum() * min_samples
        else : 
            self._min_samples = min_samples
        # if given percentage then automatically calculate the minimum bad number for each bin
        if 0 < min_bads and min_bads <= 1 :
            self._min_bads = self._MonoDf['bads'].sum() * min_bads
        else : 
            self._min_bads = min_bads
        
        self._pvalue = init_pvalue
        self._maximize_bins = maximize_bins

    @property
    def MonoDf(self) :
        return self._MonoDf
    @property
    def max_bins(self) :
        return self._max_bins
    @property
    def min_bins(self):
        return self._min_bins
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
    def pvalue(self):
        return self._pvalue
    @property
    def maximize_bins(self) :
        return self._maximize_bins        
      
    def __runTwoSampleTTest(self, mergeMethod : str, bin1_mean, bin2_mean, bin1_std, bin2_std, bin1_total, bin2_total) -> list:
        '''
        two-tailed Test
        H0 : mu_A  = mu_B
        H1 : mu_A != mu_B
        '''

        newTotal = bin1_total + bin2_total
        dof = newTotal - 2
        newMean = ((bin1_mean * bin1_total) + (bin2_mean * bin2_total)) / newTotal
        
        pooled_std = ((bin1_total - 1) * bin1_std ** 2 + (bin2_total - 1) * bin2_std ** 2) / dof
        
        if pooled_std > 0 :
            z = (bin1_mean - bin2_mean) / (pooled_std * ((1 / bin1_total) + (1 / bin2_total))) ** 0.5
        # using student's T dist to calculate p-value, when dof is large enough, it will approach to normal dist
            p_value =  2 * (1 - scipy.stats.t.cdf(abs(z), df = dof)) 
        else :
            p_value =  1
            
        # if the bins only contains bad (to avoid inf woe)
        if (bin1_mean == 1) or (bin2_mean == 1) :
            p_value += 2
        # if the bin does not contain any bads (to avoid inf woe)
        if (bin1_mean == 0) or (bin2_mean == 0) :
            p_value += 2    
        
        # if the number of the default is less than md -> add 1 to p-value
        if (bin1_mean * bin1_total < self.min_bads) or (bin2_mean * bin2_total < self.min_bads) :
            p_value += 1
        # if the number of observations is less than mo -> add 1 to p-value
        if (bin1_total < self.min_samples) or (bin2_total < self.min_samples):
            p_value += 1
            if newTotal < self.min_samples :
                p_value += 1
        # if a bin contains just one observation then set p-value = 2
        if bin1_total == 1 or bin2_total == 1 :
            if 1 in [bin1_total, bin2_total] :
                p_value = 3
        
        if mergeMethod == 'Size' :
            if (bin1_total < self.min_samples) or (bin2_total < self.min_samples):
                p_value += 1
            
            if (bin1_mean * bin1_total < self.min_bads) or (bin2_mean * bin2_total < self.min_bads) :
                p_value += 1
                
            if ((newTotal * newMean) < self.min_bads) or ((bin1_mean == 0) or (bin2_mean == 0)):
                p_value += 2
            
            if (newTotal > self.max_samples) and ((bin1_total < self.min_samples) or (bin2_total < self.min_samples)) :
                # although merging result will exceed the limitation, we expect to see a large bin rather than a tiny one
                p_value += 2
            elif newTotal > self.max_samples : 
                # make sure not to merge the two bins which will generate bins dist > max_samples
                if (bin1_total != 1) and (bin2_total != 1) and (bin1_mean not in [0,1]) and (bin2_mean not in [0,1]) :
                    p_value *= 0.01
                else :
                    p_value *= 0.1
            else :
                pass

        return [p_value, newTotal, newMean, pooled_std**0.5]
        
    
    def __calculateAdjacentPValue(self, optTable, mergeMethod) -> list:
        p_value_list = []
        num_bins = len(optTable)
        
        for i in range(num_bins - 1):
            bin1_mean = optTable.loc[i, 'mean']
            bin2_mean = optTable.loc[i + 1, 'mean']
            bin1_std  = optTable.loc[i, 'std']
            bin2_std  = optTable.loc[i + 1, 'std']
            bin1_obs  = optTable.loc[i, 'total']
            bin2_obs  = optTable.loc[i + 1, 'total']
            
            p_value_list.append(self.__runTwoSampleTTest(mergeMethod, bin1_mean, bin2_mean, bin1_std, bin2_std, bin1_obs, bin2_obs))

        return p_value_list

    def __mergeBinsOnce(self, optTable, mergeResult, mergeIndex) :
        newStart = optTable.loc[mergeIndex, '[intervalStart']
        newEnd   = optTable.loc[mergeIndex + 1, 'intervalEnd)']
        
        newTotal = mergeResult[mergeIndex, 1]
        newMean  = mergeResult[mergeIndex, 2]
        newBads  = newTotal * newMean
        newStd   = mergeResult[mergeIndex, 3]
        
        mergeDataFrame = pd.DataFrame({'[intervalStart' : [newStart],
                                       'intervalEnd)'   : [newEnd],
                                       'total' : [newTotal],
                                       'bads'  : [newBads],
                                       'mean'  : [newMean],
                                       'std'   : [newStd]})
        
        if mergeIndex == 0 :
            return pd.concat([mergeDataFrame, optTable.iloc[mergeIndex+2:]], axis = 0, ignore_index=True)
        elif mergeIndex == (len(optTable) - 2) :
            return pd.concat([optTable.iloc[:mergeIndex], mergeDataFrame], axis = 0, ignore_index=True)
        else :
            return pd.concat([optTable.iloc[:mergeIndex], mergeDataFrame, optTable.iloc[mergeIndex+2:]], axis = 0, ignore_index=True)
    
    @pvalue.setter    
    def __updatePvalue(self) -> None:
        # org_p = float(Decimal(self.pvalue))
        if self.pvalue > 0.01 :
            if (self.pvalue - 0.05 <= 0) or (self.pvalue - 0.05 >0.001):
                self._pvalue = 0.01
            else:
                self._pvalue -= 0.05
        else :
            self._pvalue *= 0.1
        
    def __OptBinning(self, monoTable, mergeMethod :str):
        df = monoTable.copy()
        
        if ((self.maximize_bins) and (len(df) <= self.max_bins)) or ((~self.maximize_bins) and (len(df) <= self.min_bins)) :
            # initial numbers of bins already less than the constraint -> return
            return df
        else :
            while (len(df) > self.min_bins) : # if num of bins exceed min_bins 
                if self.maximize_bins :
                    while len(df) > self.max_bins :
                        mergeResultMatrix = np.array(self.__calculateAdjacentPValue(df, mergeMethod)) # calculate p-value for all adjacent bin pairs
                        p_value_array = mergeResultMatrix[:, 0] # [p_value, newTotal, newMean, pooled_std**0.5]

                        # find maximum p-value index 
                        max_p_index = np.where(p_value_array == np.max(p_value_array))[0][0] # no matter the length of maximum p-value index, only get the first one
                        max_p = p_value_array[max_p_index]
                        # # find minimum p-value index 
                        if max_p > self.pvalue : # if accept null hypothesis, merge the two bins.
                            df = self.__mergeBinsOnce(optTable = df, mergeResult = mergeResultMatrix, mergeIndex = max_p_index)
                        elif max_p < self.pvalue and len(df) > self.max_bins:  # no p-value greater than threshold but yet meet the max_bins constraint -> update p-value
                            self.__updatePvalue()
                            if self.pvalue < 0.0000001 :
                                return df
                        else :
                            pass
                            
                            
                        if len(df) <= self.max_bins :
                            if mergeMethod == 'Size':
                                whileiter = 0
                                while (any(df['total'] < self.min_samples)) and (len(df) > self.min_bins) :
                                    mergeResultMatrix = np.array(self.__calculateAdjacentPValue(df, mergeMethod)) # calculate p-value for all adjacent bin pairs
                                    p_value_array = mergeResultMatrix[:, 0] # [p_value, newTotal, newMean, pooled_std**0.5]
                                    
                                    for p in -np.sort(-p_value_array) :
                                        for input_merge_index in np.where(p_value_array == p)[0] :
                                            if (df.loc[input_merge_index, 'total'] < self.min_samples) or (df.loc[input_merge_index + 1, 'total'] < self.min_samples) :
                                                df = self.__mergeBinsOnce(optTable=df, mergeResult=mergeResultMatrix, mergeIndex=input_merge_index)
                                                break
                                            else :
                                                continue
                                        break
                                    if (len(df) <= self.min_bins) or all(df['total'] >= self.min_samples) :
                                        return df
                            else :
                                return df
                    
                    return df
                else :
                    mergeResultMatrix = np.array(self.__calculateAdjacentPValue(df, mergeMethod)) # calculate p-value for all adjacent bin pairs
                    p_value_array = mergeResultMatrix[:, 0] # [p_value, newTotal, newMean, pooled_std**0.5]

                    # find maximum p-value index 
                    max_p_index = np.where(p_value_array == np.max(p_value_array))[0][0] # no matter the length of maximum p-value index, only get the first one
                    max_p = p_value_array[max_p_index]
                    # # find minimum p-value index 
                    min_p_index = np.where(p_value_array == np.min(p_value_array))[0][0] 
                    min_p = p_value_array[min_p_index]

                    if max_p > self.pvalue : # if accept null hypothesis, merge the two bins.
                        df = self.__mergeBinsOnce(optTable = df, mergeResult = mergeResultMatrix, mergeIndex = max_p_index)
                    elif max_p < self.pvalue and len(df) > self.min_bins :
                        self.__updatePvalue()
                        if self.pvalue < 0.0000001 :
                            return df
                    else:  # no p-value greater than threshold but yet meet the max_bins constraint -> update p-value
                        if (min_p > self.pvalue) and (len(df) - ((p_value_array > self.pvalue).sum()) >= self.min_bins) : 
                            # if the result that merge all the bins lower than the current p threshold still > min_bins then go ahead.
                                i_iter_adjust = 0  # each time merge the bins, the original p-value index needs to minus one the make sure the correct bins index that will be merged in the next loop
                                # index 1 and index 3 need to be merged: if index 1 merge with the index 2 bin, then the original index3 bin will be the index 2 bin in new df
                                for i, p in enumerate(p_value_array) :
                                    if p > self.pvalue :
                                        df = self.__mergeBinsOnce(optTable = df, mergeResult = mergeResultMatrix, mergeIndex = i - i_iter_adjust)
                                        i_iter_adjust += 1
                        else :
                            break
                        
            if mergeMethod == 'Size':
                whileiter = 0
                while (any(df['total'] < self.min_samples)) and (len(df) > self.min_bins) :
                    mergeResultMatrix = np.array(self.__calculateAdjacentPValue(df, mergeMethod)) # calculate p-value for all adjacent bin pairs
                    p_value_array = mergeResultMatrix[:, 0] # [p_value, newTotal, newMean, pooled_std**0.5]
                    for p in -np.sort(-p_value_array) :

                        for input_merge_index in np.where(p_value_array == p)[0] :
                            if (df.loc[input_merge_index, 'total'] < self.min_samples) or (df.loc[input_merge_index + 1, 'total'] < self.min_samples) :
                                df = self.__mergeBinsOnce(optTable=df, mergeResult=mergeResultMatrix, mergeIndex=input_merge_index)
                                break
                            else :
                                continue

                        break
                    if (len(df) <= self.min_bins) or all(df['total'] >= self.min_samples) :
                        return df
            return df            
                        
        
    def monoOptBinning(self, mergeMethod :str = 'Stats') :
        df = self.MonoDf.copy()
        
        if mergeMethod in ['Stats', 'Size', 'Chi'] :
            completedTable = self.__OptBinning(monoTable = df, mergeMethod = mergeMethod)
        else :
            raise('Wrong Merging Method : <Stats> / <Size> / <Chi>')

        return completedTable[['[intervalStart', 'intervalEnd)', 'total', 'bads', 'mean']]
    
#%%

# if __name__ == '__main__' :
#     df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
#     df['default'] = df['default'] - 1
#     import os
#     os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python/MOB')
#     from numeric.Monotone import Monotone
    
#     M = Monotone(data = df, var = 'Durationinmonth', response = 'default')
#     res = M.tuneMonotone()
    
#     O = OptimalBinning(resMonotoneTable = res, max_bins=6, min_bins=4, max_samples = 0.4, min_samples=0.05, min_bads=0.05, init_pvalue=0.4, maximize_bins = True)
#     Binres = O.monoOptBinning()
#     print(Binres)
# %%
