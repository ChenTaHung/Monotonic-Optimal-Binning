from hashlib import new
import pandas as pd
import numpy as np

class monotone :
    def __init__(self, data, metric, sign, var, response) ;
        self.data = data
        self.metric = metric
        self.sign = sign
        self.var = var
        self.response = response
        if data[response].dtypes == "object" :
            self.responseType = "cat"
        else :
            self.responseType = "num"
    
    @classmethod
    def metricFunction(cls, newVarName) :
        self.metric = newVarName
    
    def initMonoTable(self, newVarName) :
        df = self.data.copy()
        >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if callable(self.metric) : #given metric is a customized function
            self.metricFunction(newVarName = newVarName)
            df[newVarName] = self.metric()
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if self.responseType == "cat":
            df.groupby(self.sign)[self.response].agg(["count", self.metric, "sum"])
        elif self.responseType == "num" :
            
    
    def tuneMono(self, initialization : bool) :
        if initialization :
            df = self.initMonoTable()

        
        
