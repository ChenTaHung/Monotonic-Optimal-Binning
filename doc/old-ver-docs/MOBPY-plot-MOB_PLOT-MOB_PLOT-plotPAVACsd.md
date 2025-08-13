<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center style = 'font-style:italic'><strong>API Reference</p></strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.plot.MOB_PLOT.plotPAVACsd </span></h1>

_`static method`_ **MOB_PLOT.`plotPAVACsd`(CSD_Summary, figsavePath: str = None , dpi:int = 300)**

> Visualize the cumulative sum diagram (CSD) and greatest convex minorant (GCM) of the PAVA result.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/plot/MOB_PLOT.py#L66-L120)

### **Parameters** : <br>

__*CSD_Summary*__ : `pandas.DataFrame`

> The given value should be the attribute **`CSD_Summary`** in the PAVA object. You can view and call the **`CSD_Summary`** attribute after running the **[`runPAVA`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA-runPAVA.md)** method. See **[`PAVA`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA.md)** for more information.

__*figsavePath*__: `str` or `None`, _default_ = `None`

> The path to export the chart if needed.

__*dpi*__: `int`, _default_ = `300`

> Dpi controls the resolution of the exported chart if `figsavePath` is given a valid value. Higher the value of dpi, better the quality of the exported image gets.

### **Returns** : `None` but generate a `matplotlib.pyplot` chart.

**Example** :

```python
import pandas as pd
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT
#import data
df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance2.csv')
# Construct PAVA and execute the algorithm
P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc=None)
P.runPAVA()
# Plotting, input PAVA's attribute CSD_Summary
MOB_PLOT.plotPAVACsd(CSD_Summary = P.CSD_Summary)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/age-insuranceclaim-PAVA.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> Cumulative sum diagram (CSD) and Greatest convex minorant (GCM) </sub> </p>