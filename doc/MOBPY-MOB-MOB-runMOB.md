<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.MOB.MOB.runMOB </span></h1>

_`instance method`_ **MOB.`runMOB`(mergeMethod, sign = 'auto')**

> Execute the MOB algorithm.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L182-L238)

### **Parameters** : <br>

**_mergeMethod_**: `str`, **{'Size', 'Stats'}**

> Specify whether a `bins size base` or a `statistical base` for the MOB algorithm to conducting optimal binnig. For more detail, please see the [**`README.md`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/README.md) documentation.


**sign**: `str`, **{'+', '-', 'auto'}**, _default_ = `'auto'`

> The sign specify the expectation of the correlation between `response` variable and the `var` variable. If set to `'+'`, then it means that the `response` variable and the `var` variable shows a positive correlation, and `'-'` for a negative correlation. If set to `'auto'` then the program will calculate the Pearson Correlation to decide the input value. See [**`response`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB.md), [**`var`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB.md) for more information.

### **Returns** : `pandas.DataFrame`

> Returns the final optimal binning result dataframe. Shows the Bins Summary. For more detail, please see the [**`README.md`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/README.md) documentation.

**Example** :

```python
# run the MOB algorithm to discretize the variable 'Creditamount'.
MOB_ALGO = MOB(data = df, var = 'Creditamount', response = 'default', exclude_value = None) 
# Set Binning Constraints (Must-Do!)
MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                max_samples = 0.4, min_samples = 0.05, 
                                min_bads = 0.05, 
                                init_pvalue = 0.4, 
                                maximize_bins=True)
# mergeMethod = 'Size' means to run MOB algorithm under bins size base
SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/Durationinmonth%20bins%20summary.png' alt = 'Image' style = 'width: 1000px'/></p>


<h3><strong> Attributes : </strong></h3>

[**`finishBinningTable`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L197-L200)

> The table that shows the binning result for the data that has precluded the missing values and exclude_values.

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)