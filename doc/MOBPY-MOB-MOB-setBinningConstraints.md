<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.MOB.MOB.setBinningConstraints </span></h1>

_`instance method`_ **MOB.`setBinningConstraints`(max_bins :int = 6, min_bins :int = 4, max_samples = 0.4, min_samples = 0.05, min_bads = 0.05, init_pvalue: float = 0.4, maximize_bins :bool = True)**

> Set the detail limitation of the binning process to obtain the optimal binning result. This is a must-do step before start running the MOB algorithm.


[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L159-L167)

### **Parameters** : <br>

__*max_bins*__ : `int`, _defualt_ = `6`

> A constraint that limits the **maximum** bins number for the binning result.

__*min_bins*__ : `int`, _defualt_ = `4`

> A constraint that limits the **minimum** bins number for the binning result.

__*max_samples*__ : `int` or `float`, _defualt_ = `0.4`

> A constraint that limits the **maximum** sample size (sample population) for each bin. If set to a value greater equal than 1, then the maximum sample will be set to the number, otherwise, if set to value between 0 and 1, then the maximum samples size will the porportion of the total samples, for example, 0.4 means 40% of the totals observations.

__*min_samples*__ : `int` or `float`, _defualt_ = `0.05`

> A constraint that limits the **minimum** sample size (sample population) for each bin. Values enter logic is same as `max_samples`. 

__*min_bads*__ : `int` or `float`, _defualt_ = `0.05`

> A constraint that limits the **minimum** positive events ([see response parameter](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB.md)) for each bin. Values enter logic is same as `max_samples`. 

__*init_pvalue*__ : `float`, _defualt_ = `0.4`

> The initial p-value for the hypothesis testing threshold to decide whether merge the adjacent bins or not. The p-value will iteratively decrease to set a stricter threshold in order to meet the bins number limitation.

__*maximize_bins*__ : `bool`, _defualt_ = `True`

> A user preference to make the merging process in the algorithm stop earlier when it can remain an optimal binning results while maximizing the number of bins. Note that when [**`mergeMethod`**]() is set to `'Size'` may overwrite the constraints.

### **Returns**: `None`


**Example** :

```python
```python
# run the MOB algorithm to discretize the variable 'Creditamount'.
MOB_ALGO = MOB(data = df, var = 'Creditamount', response = 'default', exclude_value = None) 
# Set Binning Constraints (Must-Do!)
MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                max_samples = 0.4, min_samples = 0.05, 
                                min_bads = 0.05, 
                                init_pvalue = 0.4, 
                                maximize_bins=True)
```

<br>

<h3><strong> Attributes : </strong></h3>

[**`constraintsStatus`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L167)

> The initial value of **`constraintsStatus`** is `False`. Once the user had run the method *`setBinningConstraints`()*, the value of the attribute will be set to `True` to prevent a Exception error while executing the [__*`runMOB`*__](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-runMOB.md) method.


<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)