<h1><strong><p align = center> MOBPY Documentation </p></strong></h1>

<h2><p  align=center style = 'font-style:italic'><strong>API Reference</p></strong></p></h2>

<h1><span style = 'font-size:smaller'> MOBPY.plot.MOB_PLOT.plotBinsSummary </span></h1>

_`static method`_ **MOB_PLOT.`plotBinsSummary`(monoOptBinTable, var_name, bar_fill = 'skyblue', bar_alpha = 0.5, bar_width = 0.5, bar_text_color = 'darkblue', line_color = 'orange', line_width = 3, dot_color = 'red', dot_size = 80, annotation_font_weight = 'bold', figsavePath: str = None , dpi:int = 300)**

> Visualize the binning Summary dataframe.

[**[source]**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/src/MOBPY/plot/MOB_PLOT.py#L7-L62)

### **Parameters** : <br>

**_monoOptBinTable_** : `pandas.DataFrame`

> The return data frame from [***`MOB.runMOB`*()**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB-runMOB.md). 
>
> Input example :

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/Durationinmonth%20bins%20summary.png' alt = 'Image' style = 'width: 800px'/></p>

__*var_name*__: `str`

> The name of the [**`var`**](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-MOB-MOB.md) variable to show in the chart.

__*bar_fill*__: `str`, _default_ = `'skyblue'`

> The color used to fill the bars in the bar chart layers. See [matplotlib:List of named colors](https://matplotlib.org/stable/gallery/color/named_colors.html) for more information.

__*bar_alpha*__: `float`, _default_ = `0.5`

> The color transparency of the bars in the barchart layers. See [matplotlib:Fill Between and Alpha](https://matplotlib.org/3.3.4/gallery/recipes/fill_between_alpha.html) for more information.

__*bar_width*__: `float`, _default_ = `0.5`

> The bar width of the bars in the barchart layers. See [matplotlib.pyplot.bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) for more information.

__*bar_text_color*__: `str`, _default_ = `'darkblue'`

> The color of the text showing the information of the bar charts, which respresent the population proportion of the bin.

__*line_color*__: `str`, _default_ = `'orange'`

> The color of the line in the line chart layer. See [matplotlib:Specifying colors](https://matplotlib.org/stable/tutorials/colors/colors.html) for more information.

__*line_width*__: `float`, _default_ = `3.0`

> The width of the line in the line chart layer. See [matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html) for more information.

__*dot_color*__: `str`, _default_ = `'red'`

> The color of the point in the scatter chart layer. See [matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) for more information.

__*dot_size*__: `float`, _default_ = `80.0`

> The size of the point in the scatter chart layer. See [matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html) for more information.

__*annotation_font_weight*__: `str`, _default_ = `'bold'`

> The style of the font for the text in the entire chart, including the text showing the bar chart information (population proportion) and also the *Bad Rate* (positive term / total bins sample). [matplotlib:Text properties and layout](https://matplotlib.org/stable/tutorials/text/text_props.html) 

__*figsavePath*__: `str` or `None`, _default_ = `None`

> The path to export the chart if needed.

__*dpi*__: `int`, _default_ = `300`

> Dpi controls the resolution of the exported chart if `figsavePath` is given a valid value. Higher the value of dpi, better the quality of the exported image gets.

### **Returns** : `None` but generate a `matplotlib.pyplot` chart.

**Example** :

```python
MOB_ALGO = MOB(data = df, var = 'Creditamount', response = 'default', exclude_value = None) 
# Set Binning Constraints (Must-Do!)
MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                max_samples = 0.4, min_samples = 0.05, 
                                min_bads = 0.05, 
                                init_pvalue = 0.4, 
                                maximize_bins=True)
# mergeMethod = 'Size' means to run MOB algorithm under bins size base
SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size')

MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/Durationinmonth-Size.png' alt = 'Image' style = 'width: 1200px'/></p>


<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)