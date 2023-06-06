import pandas as pd
import numpy as np
import scipy.stats

class OptimalBinning :
    def __init__(self, resMonotoneTable, max_bins :int, min_bins:int, max_samples, min_samples, min_bads, init_pvalue :float, maximize_bins :bool) :
        self.MonoDf = resMonotoneTable
        self.max_bins = max_bins
        self.min_bins = min_bins
        # if given percentage then automatically calculate the maximum number of observation for each bin
        if 0 < max_samples and max_samples <= 1 :
            self.max_samples = self.MonoDf['total'].sum() * max_samples
        else : 
            self.max_samples = max_samples
        # if given percentage then automatically calculate the minimum number of observation for each bin
        if 0 < min_samples and min_samples <= 1 :
            self.min_samples = self.MonoDf['total'].sum() * min_samples
        else : 
            self.min_samples = min_samples
        # if given percentage then automatically calculate the minimum bad number for each bin
        if 0 < min_bads and min_bads <= 1 :
            self.min_bads = self.MonoDf['bads'].sum() * min_bads
        else : 
            self.min_bads = min_bads
        
        self.pvalue = init_pvalue
        self.maximize_bins = maximize_bins
        
      
    def _runTwoSampleTTest(self, mergeMethod, bin1_mean, bin2_mean, bin1_std, bin2_std, bin1_total, bin2_total) -> list:
        '''
        two-tailed Test
        H0 : mu_A  = mu_B
        H1 : mu_A != mu_B

        merging by p-value :
            Compute z-test p-value for all adjacent bins, 
                if the number of defaults is less than mo or the number of observations is less than mo 
                    add 1 to p-value, 
                if a bin contains just one observation 
                    then set p-value=2.
            Merge two bins with the highest p-value and repeat(10) till all p-values are below the critical p-value.
        '''
        newTotal = bin1_total + bin2_total
        dof = newTotal - 2
        newMean = ((bin1_mean * bin1_total) + (bin2_mean * bin2_total)) / newTotal
        
        pooled_std = ((bin1_total - 1) * bin1_std ** 2 + (bin2_total - 1) * bin2_std ** 2) / dof
        
        if mergeMethod in ['MFB','SFB'] :
            if pooled_std > 0 :
                z = (bin1_mean - bin2_mean) / (pooled_std * ((1 / bin1_total) + (1 / bin2_total))) ** 0.5
            # using student's T dist to calculate p-value, when dof is large enough, it will approach to normal dist
                p_value =  2 * (1 - scipy.stats.t.cdf(abs(z), df = dof)) 
            else :
                p_value =  2
                
            if (bin1_total < self.min_samples) or (bin2_total < self.min_samples):
                p_value += 1
                
            if (bin1_mean * bin1_total < self.min_bads) or (bin2_mean * bin2_total < self.min_bads) :
                p_value += 2

            if (newTotal * newMean) < self.min_bads :
                p_value += 3
            
            if mergeMethod == 'SFB' :
                if newTotal >= self.max_samples : 
                    # make sure not to merge the two bins which will generate bins dist > max_samples
                    p_value *= -1
                
                if newTotal < self.min_samples :
                    p_value += 3

        return [p_value, newTotal, newMean, pooled_std**0.5]
        
    
    def _calculateAdjacentPValue(self, optTable, mergeMethod) -> list:
        p_value_list = []
        num_bins = len(optTable)
        
        for i in range(num_bins - 1):
            bin1_mean = optTable.loc[i, 'mean']
            bin2_mean = optTable.loc[i + 1, 'mean']
            bin1_std  = optTable.loc[i, 'std']
            bin2_std  = optTable.loc[i + 1, 'std']
            bin1_obs  = optTable.loc[i, 'total']
            bin2_obs  = optTable.loc[i + 1, 'total']
            
            p_value_list.append(self._runTwoSampleTTest(mergeMethod, bin1_mean, bin2_mean, bin1_std, bin2_std, bin1_obs, bin2_obs))

        return p_value_list

    def _mergeBins(self, optTable, mergeResult, mergeIndex) :
        newStart = optTable.loc[mergeIndex, 'start']
        newEnd   = optTable.loc[mergeIndex + 1, 'end']
        
        newTotal = mergeResult[mergeIndex, 1]
        newMean  = mergeResult[mergeIndex, 2]
        newBads  = newTotal * newMean
        newStd   = mergeResult[mergeIndex, 3]
        
        mergeDataFrame = pd.DataFrame({'start' : [newStart],
                                       'end'   : [newEnd],
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
        
    def _updatePvalue(self) -> None:
        if self.pvalue <= 0.01 :
            if self.pvalue - 0.05 <= 0 :
                self.pvalue = 0.01
            else:
                self.pvalue -= 0.05
        else :
            self.pvalue *= 0.1
            
    def _OptBinning(self, monoTable, mergeMethod :str):
        df = monoTable.copy()
        
        if len(df) <= self.max_bins :
            return df
        else :
            # assign a initial p_value list contains all 1 to start the while loop.
            p_value_array = np.repeat(1, len(df) - 1)
            while (p_value_array > self.pvalue).sum() > 0 :

                # calculate p-value for all adjacent bin pairs
                mergeResultMatrix = np.array(self._calculateAdjacentPValue(df, mergeMethod))
                p_value_array = mergeResultMatrix[:, 0] # [p_value, newTotal, newMean, pooled_std**0.5]

                # find maximum p-value index 
                max_p_index = np.where(p_value_array == np.max(p_value_array))[0][0] # no matter the length of maximum p-value index, only get the first one
                max_p = p_value_array[max_p_index]

                if max_p > self.pvalue : # if accept null hypothesis, merge the two bins.
                    df = self._mergeBins(optTable = df, mergeResult = mergeResultMatrix, mergeIndex = max_p_index)
                    
                    if len(df) - ((p_value_array > self.pvalue).sum()) >= self.min_bins : 
                        # if the result that merge all the bins lower than the threshold still > min_bins then go ahead.
                        continue
                    else:
                        if self.maximize_bins :
                            if (df['total'].min() < self.min_samples) and (len(df) > self.min_bins) :
                                continue
                            else :
                                if len(df) <= self.max_bins :
                                    break
                                else :
                                    continue
                        else :
                            if len(df) <= self.min_bins :
                                break
                            else :
                                continue

                elif (max_p <= self.pvalue) and (len(df) > self.max_bins) : 
                    # if no p-value exceeds the p threshold, but bins cnt is greater than the maximum limitation
                    self._updatePvalue()
                else :
                    break
            
            return df
        
    def monoOptBinning(self, mergeMethod :str = 'MFB') :
        df = self.MonoDf.copy()
        
        if mergeMethod in ['MFB', 'SFB', 'CMB'] :
            completedTable = self._OptBinning(monoTable = df, mergeMethod = mergeMethod)
        else :
            raise('Wrong Merging Method : <MFB> / <SFB> / <CMB>')

        return completedTable[['start', 'end', 'total', 'bads', 'mean']]
    
#%%

# if __name__ == '__main__' :
#     df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
#     df['default'] = df['default'] - 1
#     import os
#     os.chdir('/Users/chentahung/Desktop/git/mob-py/src/main/python/MOB')
#     from numeric.Monotone import Monotone
    
#     M = Monotone(data = df, var = 'Durationinmonth', response = 'default')
#     res = M.tuneMonotone()
    
#     O = OptimalBinning(resMonotoneTable = res, missingExist=False, excludeValueExist=False, max_bins=6, min_bins=4, max_samples = 0.4, min_samples=0.05, min_bads=0.05, init_pvalue=0.4)
#     Binres = O.monoOptBinning()
#     print(Binres)
# %%
