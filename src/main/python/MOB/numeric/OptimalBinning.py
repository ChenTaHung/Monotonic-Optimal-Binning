import pandas as pd
import numpy as np
import scipy.stats


class OptimalBinning :
    def __init__(self, resMonotoneTable, missingExist, excludeValueExist, max_bins, min_bins, max_samples, min_samples, min_bads, init_pvalue) :
        self.MonoDf = resMonotoneTable
        self._isNaExist = missingExist
        self._isExcValueExist = excludeValueExist
        self.max_bins = max_bins
        self.min_bins = min_bins
        # if given percentage then automatically calculate the maximum number of observation for each bin
        if 0 < max_samples and max_samples <= 1 :
            self.max_samples = self.MonoDf["total"].sum() * max_samples
        else : 
            self.max_samples = max_samples
        # if given percentage then automatically calculate the minimum number of observation for each bin
        if 0 < min_samples and min_samples <= 1 :
            self.min_samples = self.MonoDf["total"].sum() * min_samples
        else : 
            self.min_samples = min_samples
        # if given percentage then automatically calculate the minimum bad number for each bin
        if 0 < min_bads and min_bads <= 1 :
            self.min_bads = self.MonoDf["bad"].sum() * min_bads
        else : 
            self.min_bads = min_bads
        
        self.pvalue = init_pvalue
        
      
    def runTTest(self):
        '''
        two-tailed Test
        H0 : mu_A  = mu_B
        H1 : mu_A != mu_B
        t = (mean_A - mean_B) / sqrt((std_A^2/n_A) + (std_B^2/n_B))
        p_value = scipy.stats.norm.sf(abs(x)) * 2
        '''
    
    def _updatePvalue(self) -> None:
        if self.pvalue <= 0.01 :
            if self.pvalue - 0.05 <= 0 :
                self.pvalue = 0.01
            else:
                self.pvalue -= 0.05
        else :
            self.pvalue *= 0.1
    
    def checkBinsCnt(self, optTable) :
        return optTable.shape[0]
    
    def checkCurBinsCnt(self, optTable) :
        if self._isNaExist and self._isExcValueExist :
            curBinsCnt = checkBinsCnt(optTable) + 2
        elif (self._isNaExist and ~self._isExcValueExist) or (~self._isNaExist and self._isExcValueExist) :
            curBinsCnt = checkBinsCnt(optTable) + 1
        elif ~self._isNaExist and ~self._isExcValueExist:
            curBinsCnt = checkBinsCnt(optTable)
        
        return curBinsCnt
    
    def summarizeBins(self, FinalOptTable):
        df = FinalOptTable.copy()
        
        df = df[["start", "end", "total", "bad", "mean"]].rename(columns = {"total": "nsamples", "bad" : "bads", "mean" : "bad_rate"})
        df["dist_obs"] = df["nsamples"] / df["nsamples"].sum()
        df["dist_bads"] = df["bads"] / df["bads"].sum()
        df["goods"] = df["nsamples"] - df["bads"]
        df["dist_goods"] = df["goods"] / df["goods"].sum()
        df["woe"] = np.log(df["dist_goods"]/df["dist_bads"])
        df["iv_grp"] = (df["dist_goods"] - df["dist_bads"]) * df["woe"]
        
        return df
        
            
    def MFB(self):
        #check initial bins number
        if checkCurBinsCnt(optTable = self.MonoDf) <= self.max_bins :
            summarizeBins(self.MonoDf) # Exit
        else :
            optTable = self.MonoDf
    
    def mergeBins(self, mergeTarget : str, mergeMethod = "MFB" :str) :
        df = self.df.copy()
        
                
        return woe_summary