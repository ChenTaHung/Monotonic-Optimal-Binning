<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center><strong style = 'font-style:italic'>API Reference</strong></p></h2>


<h1><span style = 'font-size:smaller'> MOBPY.MOB.MOB </span></h1>

`MOBPY.MOB.MOB`([data, var, response, exclude_value])

*`class`* **MOBPY.MOB.`MOB`(data = None, var, response, exclude_value = None)**

> Construct the object `MOBPY`.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L6-L79)

### **Parameters** : <br>

**_data_** : `pandas.DataFrame`

> The pandas dataframe that at least include the `var` column to be discritzed and the `response` column, which is normally the dependent variable column in a statistical model.

**_var_** : `str`

> Target variable that will run through the MOB alogirthm. The column name of the variable that is decided to be discreitzed.


**_response_** : `str`

> The variable to calculate the sorting and merging metric for each bin. The column name of the variable that is used to be the dependent varaible in statistical models or the label column in machine learning models. Be sure that the distribution follows a Bernoulli distribution, with a datatype of `int` or `float`, and make 1 represent the positive event, 0 for the negative event.


__*exclude_value*__ : `int`, `float`, `list` or `None`, _default_ = `None`

> The value that will be preclude before start discretize the `var` variable. It can either enter a integer, a float or specify a list contains the exclude values.

### **Returns** : `object class = MOBPY.MOB`

**Example** :

```python
from MOBPY.MOB import MOB

MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = None)
```

<br>

<h3><strong> Methods : </strong></h3>

[**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)([max_bins, min_bins, max_samples, min_samples, min_bads, init_pvalue, maximize_bins])

> Set the limitations of the optimal binning results for the MOB algorithm.

[**`runMOB`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-runMOB.md)([mergeMethod, sign])

> Execute the MOB algorithm.

<br>

<h3><strong> Attributes : </strong></h3>

[**`max_bins`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L124-L126)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`min_bins`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L131-L133)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`max_samples`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L139-L140)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`min_samples`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L143-L145)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`min_bads`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L147-L149)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`maximize_bins`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L151-L153)

> See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`constraintsStatus`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L100-L102)

> A boolean value shows that whether the user has set the binning constraints. See [**`setBinningConstraints`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)

[**`finishBinningTable`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/MOB.py#L155-L157)

> The table that shows the binning result for the data that has precluded the missing values and exclude_values.

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)