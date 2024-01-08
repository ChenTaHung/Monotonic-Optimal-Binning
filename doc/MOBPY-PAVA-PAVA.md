<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>


<h1><span style = 'font-size:smaller'> MOBPY.PAVA.PAVA </span></h1>

`MOBPY.PAVA.PAVA`(data, var, response, metric, add_var_aggFunc, exclude_value)

*`class`* **MOBPY.PAVA.`PAVA`(data, var :str, response : str, metric : str, add_var_aggFunc : dict = None, exclude_value = None)**

> Construct the object `PAVA`.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L8-L84)

### **Parameters** : <br>

__*data*__ : `pandas.DataFrame`

> The pandas dataframe that at least include the `var` column to be discritzed and the `response` column. If `add_var_aggFunc` is given, make sure the additional variable columns are also included in the data.

__*var*__ : `str`

> Target variable that will run through the PAVA. The column name of the variable that is decided to be discreitzed.

**_response_** : `str`

> The variable to calculate the sorting and merging metric for each bin. Be sure that the column `response` belongs to one of the following type, a float or a interger.

**_metric_** : `str`, **{'count', "mean", 'sum', 'std', 'var', 'min','max', 'ptp'}**

> The statistical metric that PAVA will use to determine the binning result. 
> 
> Acceptable values :
> 1. count : numbers of observation
> 2. mean : mean, average
> 3. sum : summation
> 4. std : standard deviation
> 5. var : variance
> 6. min : minimum value
> 7. max : maximum value
> 8. ptp : peak to peak, equivalent to range(maximum - minimum), see [numpy.ptp](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html).

__*add_var_aggFunc*__: `dict`, _default_ = `None`

> A dicitionary of additional variables comes with their statistics metric to show in the final PAVA result table. A valid input follows the variable column name as the key, and the desire aggregation as the values.
> 
>  The acceptable aggration values are :
> 
> - function
> - string function name
> - list of functions and/or function names, e.g.`[np.sum, 'mean']`
>
> See [pandas.DataFrame.agg](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) for more information

__*exclude_value*__ : `int`, `float`, `list` or `None`, _default_ = `None`

> The value that will be preclude before start discretize the `var` variable. It can either enter a integer, a float or specify a list contains the exclude values.


### **Returns** : `object class = MOBPY.PAVA`

**Example** :

```python
import pandas as pd
from MOBPY.PAVA import PAVA

# importing data
df = pd.read_csv('~/mob-py/data/insurance2.csv')

# construct PAVA object.
P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean')
```

<br>

<h3><strong> Methods : </strong></h3>

[**`runPAVA`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA-runPAVA.md#L175-L254)([sign])

> Execute the PAV algorithm (PAVA).

[**`applyPAVA`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-PAVA-PAVA-applyPAVA.md#L256-L268)([var_column, assign])

> Apply the MOB result to the given variable column (`var_column``) and create a new pandas.Series.

<br>

<h3><strong> Attributes : </strong></h3>

[**`PAV_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L118-L120)

> The dataset showing the continous interval information, along with the additional variable aggregation result and the corresponding metric values.

[**`CSD_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L114-L116)

> The cumulative metric recorded through the PAVA process, it also shows how PAVA roll back to get the monotonic metric to comply with the restriction according to the [reference](https://repository.tudelft.nl/islandora/object/uuid:5a111157-1a92-4176-9c8e-0b848feb7c30) **`Eq. 1.1`**.

[**`GCM_Summary`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L110-L112)

> Greatest convex minorant summary result. It shows the interval and the calculated metric value.

[**`OrgDataAssignment`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/PAVA.py#L106-L108)

> The original dataset with chosen `var` and `response` columns, along with the assignment of the "starting" value of the given interval from the PAVA binning result and metrics.



<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)