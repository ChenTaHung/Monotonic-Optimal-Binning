<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.PAVA.PAVA.runPAVA </span></h1>

_`instance method`_ **MOB.`runMOB`(sign = 'auto')**

> Execute the MOB algorithm.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L175-L254)

### **Parameters** : <br>

**sign**: `str`, **{'+', '-', 'auto'}**, _default_ = `'auto'`

> The sign specify the expectation of the correlation between the metric of the `response` variable and the `var` variable. If set to `'+'`, then it means that the `response` variable's metric values and the `var` variable shows a positive correlation, and `'-'` for a negative correlation. If set to `'auto'` then the program will calculate the Pearson Correlation to decide the input value. See [**`response`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA.md), [**`var`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA.md) for more information.

### **Returns** : `None`

> **`runPAVA`** will store 4 attributes into the `PAVA` object.


- > [**`PAV_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L118-L120): The dataset showing the continous interval information, along with the additional variable aggregation result and the corresponding metric values.

- > [**`CSD_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L114-L116): The cumulative metric recorded through the PAVA process, it also shows how PAVA roll back to get the monotonic metric to comply with the restriction according to the [reference](https://repository.tudelft.nl/islandora/object/uuid:5a111157-1a92-4176-9c8e-0b848feb7c30) **`Eq. 1.1`**.

- > [**`GCM_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L110-L112): Greatest convex minorant summary result. It shows the interval and the calculated metric value.

- > [**`OrgDataAssignment`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L106-L108): The original dataset with chosen `var` and `response` columns, along with the assignment of the "starting" value of the given interval from the PAVA binning result and metrics.

**Example** :

```python
import pandas as pd
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT

df = pd.read_csv('~/data/insurance2.csv')

P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc=None)
P.runPAVA()

# four attributes
print(P.PAV_Summary)
print(P.CSD_Summary)
print(P.GCM_Summary)
print(P.OrgDataAssignment)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/age-on-insurance-PAVA-summary.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> PAVA.PAV_Summary output </sub> </p>

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)