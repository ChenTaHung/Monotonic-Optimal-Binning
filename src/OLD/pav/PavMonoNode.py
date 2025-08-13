import numpy as np
from typing import Union

class PavMonoNode:
    
    pre = None
    next = None
    
    def __init__(self, value, stats_array, metric):
        '''
        startValue : original value
        endValue : ɸ(x) -> values that violates the convex (non-convex points will be transformed into the previous convex point)
        single_node_array : <array> ['count', 'sum', 'std', 'max', 'min']
        
        '''
        self._startValue = value
        self._endValue = value
        self._metric = metric
        self.stats_array = stats_array
    
    @property
    def startValue(self) -> Union[float, int]: 
        return self._startValue
    
    @property
    def endValue(self) -> Union[float, int]: 
        return self._endValue
    
    @property
    def metric(self) -> str :
        return self._metric
    
    @property
    def count(self) -> int :
        return self.stats_array[0]
    
    @property
    def sum(self) -> float :
        return self.stats_array[1]
    
    @property
    def std(self) -> float :
        return self.stats_array[2]
    
    @property
    def max(self) -> float :
        return self.stats_array[3]
    
    @property
    def min(self) -> float :
        return self.stats_array[4]
    
    @property
    def mean(self) -> float :
        return self.sum / self.count
    
    @property
    def ptp(self) -> float :
        return self.max - self.min
    
    @property
    def var(self) -> float :
        return self.std ** 2
    

    @property
    def stats(self) -> Union[float, int]:
        '''
        acceptable inputs : { count | mean | sum | std | var | min | max | ptp }
        '''
        if self.metric == 'count' :
            return self.count
        
        elif self.metric == 'mean' :
            return self.mean
        
        elif self.metric == 'sum' :
            return self.sum
        
        elif self.metric == 'std' :
            return self.std
        
        elif self.metric == 'var' :
            return self.var
        
        elif self.metric == 'min' :
            return self.min
        
        elif self.metric == 'max' :
            return self.max
        
        elif self.metric == 'ptp' :
            return self.ptp

        else :
            return np.nan
        
        
        
    def update(self, nextValue, nextCount, nextSum, nextStd, nextMax, nextMin) -> None :
        '''
            startValue | endValue | count | sum | std | max | min 
            ------------------------------------------------------
                       |          |       |     |     |     |                
        '''
        
        if (self.count + nextCount) == 2 :
            newStd = np.std(np.array(self.mean, (nextSum/nextCount))) # newMean = newSum / newCount
        else :
            newStd = np.sqrt(((self.count * (self.std ** 2)) + (nextSum * (nextStd ** 2))) / (self.count + nextCount - 1))
        
        self.stats_array[0] = self.count + nextCount
        self.stats_array[1] = self.sum + nextSum
        self.stats_array[2] = newStd
        self.stats_array[3] = np.max([self.max, nextMax])
        self.stats_array[4] = np.min([self.min, nextMin])
        
        self._endValue = nextValue
            

