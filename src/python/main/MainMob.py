
def main(data, xList, y) :
    if needMetric == True :
        customMetric = defineMetric()
    for x in xList :
        # discretize variable into montonic trend
        mono = mob.monotone(data = dataframe, var = x, response = y, metric = customMetric)
        monoTable = mono.tuneMonotone(monoTarget = )
        
        # pool values from adjacent bins under given constraints and form new bins
        binning = mob.optimalBinning(MonotonicTable, max_bins, min_bins, max_dist, min_dist, initial_pvalue)
        res = binning.merge(mergeTarget = , mergeType = <"SFB"|"MFB"> )
        
        if Visualization == True :
            plot(res)
        
if __name__ == '__main__' :
    main()    
            