<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.PAVA.PAVA.applyPAVA </span></h1>

_`instance method`_ **MOB.`applyPAVA`(var_column : pd.Series, assign :str = 'interval')**

> Apply the MOB result to the given variable column (`var_column`) and create a new pandas.Series.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L256-L268)

### **Parameters** : <br>

__*var_column*__ : `pandas.Series`

> The column that need to apply the dicretization result generated from [***`PAVA.runPAVA`*()**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA-runPAVA.md). 

__*assign*__ : `str`, **{'interval', 'start', 'end'}** _default_ = `'interval'`

> The representation way of the output.
> 
> - interval : `<str>` : assign the interval as the value display. e.g. **[value_start, value_end)**
> - start : `<float>` : assign the starting value of the interval.
> - end : `<float>` : assign the closing value of the interval.

### **Returns** : `pandas.Series`

**Example** :

```python
import pandas as pd
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT

df = pd.read_csv('~/data/insurance2.csv')

P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc=None)
P.runPAVA()

# apply PAVA result
print(P.applyPAVA(df['age'])) #default = 'interval'
print(P.applyPAVA(df['age'], 'start'))
print(P.applyPAVA(df['age'], 'end'))

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
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyPAVA-interval-res.png" width="300" >
        </td>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyPAVA-start-res.png" width="300" >
        </td>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/applyPAVA-end-res.png" width="300" >
        </td>
      </tr>
    </tbody>
  </table>
</div>

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)