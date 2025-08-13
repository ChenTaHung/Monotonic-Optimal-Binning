<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.MOB.MOB.applyMOB </span></h1>

_`instance method`_ **MOB.`applyMOB`(var_column : pd.Series, assign :str = 'interval')**

> Apply the MOB result to the given variable column (`var_column`) and create a new pandas.Series.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L258-L273)

### **Parameters** : <br>

__*var_column*__ : `pandas.Series`

> The column that need to apply the dicretization result generated from [***`MOB.runMOB`*()**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-runMOB.md). 

__*assign*__ : `str`, **{'interval', 'start', 'end'}**, _default_ = `'interval'`

> The representation way of the output.
> 
> - interval : `<str>` : assign the interval as the value display. e.g. **[value_start, value_end)**
> - start : `<float>` : assign the starting value of the interval.
> - end : `<float>` : assign the closing value of the interval.

### **Returns** : `pandas.Series`

**Example** :

```python
df = pd.read_csv('~/mob-py/data/german_data_credit_cat.csv')
df['default'] = df['default'] - 1
MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = None) 
# Set Binning Constraints (Must-Do!)
MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                max_samples = 0.4, min_samples = 0.05, 
                                min_bads = 0.05, 
                                init_pvalue = 0.4, 
                                maximize_bins=True)
# mergeMethod = 'Size' means to run MOB algorithm under bins size base
SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')
# apply MOB binning result to the column : df['Durationinmonth']
# the default setting is set as 'interval'
print(MOB_ALGO.applyMOB(df['Durationinmonth']))
print(MOB_ALGO.applyMOB(df['Durationinmonth'], 'start'))
print(MOB_ALGO.applyMOB(df['Durationinmonth'], 'end'))
```

<div>
  <table >
    <thead>
      <tr>
        <th style="text-align: center;">assign = 'interval'</th>
        <th style="text-align: center;">assign = 'start'</th>
        <th style="text-align: center;">assign = 'end'</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center;">show interval </td>
        <td style="text-align: center;">show starting value</td>
        <td style="text-align: center;">show closing value</td>
      </tr>
      <tr>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyMOB-interval-res.png" width="300" >
        </td>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyMOB-start-res.png" width="300" >
        </td>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyMOB-end-res.png" width="300" >
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)