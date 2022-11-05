import pandas as pd
import numpy as np
from mobFunc import MobFunc

class OptimalBinning(MobFunc) :
    def __init__(self, MonotonicTable, max_bins, min_bins, max_dist, min_dist, max_samples, min_samples, min_bads, init_pvalue, exclude_value = None) :
        super().__init__()
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