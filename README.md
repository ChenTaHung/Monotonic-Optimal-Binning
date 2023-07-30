<h1><p align = 'center'><strong> Monotonic-Optimal-Binning </strong> </p></h1>
<h3><p align = 'center'><strong> Python implementation (MOBPY) </strong> </p></h3>

[**MOB**](https://pypi.org/project/MOBPY/) is a statistical approach designed to transform continuous variables into categorical variables in a way that ensures both optimality and monotonicity. The project consists of two main modules: **`MOB`** and **`PAVA`**. The `MOB` module is responsible for achieving monotonic optimal binning, while the `PAVA` module utilizes the pool adjacent violators algorithm.

In MOB, we have extended the functionality to allow users to merge bins either based on `statistics` or `bin size`, giving them greater control over the binning process. On the other hand, PAVA offers flexibility by handling multiple statistics while MOB can only deal with average so far, and it also enables the display of other variables with given statistics in the final results.

This Python-based project empowers programmers to obtain precise and tailored discretization results, aligning with their specific expectations for monotonic categorical variables.<br>

<h2><strong> Installation </strong></h2>

```bash
python3 -m pip install MOBPY
```

<h2><strong> Usage </strong></h2>


**_MOB Example_**:

```python
import pandas as pd
from MOBPY.MOB import MOB


if __name__ == '__main__' :
    # import the testing datasets
    df = pd.read_csv('/data/german_data_credit_cat.csv')
    
    # Original values in the column are [1,2], make it into 1 representing the positive term, and 0 for the other one.
    df['default'] = df['default'] - 1

    # run the MOB algorithm to discretize the variable 'Durationinmonth'.
    MOB_ALGO = MOB(data = df, var = 'Durationinmonth', response = 'default', exclude_value = None)
    # A must-do step to set the binning constraints.
    MOB_ALGO.setBinningConstraints( max_bins = 6, min_bins = 3, 
                                    max_samples = 0.4, min_samples = 0.05, 
                                    min_bads = 0.05, 
                                    init_pvalue = 0.4, 
                                    maximize_bins=True)
    # execute the MOB algorithm.
    SizeBinning = MOB_ALGO.runMOB(mergeMethod='Size') # Run under the bins size base.

    StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats') # Run under the statistical base. 
    
```


The `runMOB` method will return a `pandas.DataFrame` which shows the binning result of the variable and also the WoE summary information for each bin. 

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/Durationinmonth-bins-summary.png' alt = 'Image' style = 'width: 800px'/></p>

<p align = 'center'> <sub> MOB algorithm binning result </sub> </p>

And after we receive the binning result dataframe, we can plot it by using [`MOBPY.plot.MOB_PLOT.plotBinsSummary`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/tree/main/doc/MOBPY-plot-MOB_PLOT-MOB_PLOT-plotBinsSummary.md) to visualize the binning summary result.

```python
from MOBPY.plot.MOB_PLOT import MOB_PLOT

# plot the bin summary data.
print('Bins Size Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')

print('Statisitcal Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = 'Durationinmonth')
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/Durationinmonth-Size.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> MOB visualization example </sub> </p>

**_PAVA Example_**:

```python
import pandas as pd
from MOBPY.PAVA import PAVA
from MOBPY.plot.MOB_PLOT import MOB_PLOT

#import testing dataset
df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/insurance2.csv')
# construct the PAVA object
P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc = {'bmi':'mean', 'smoker':'sum', 'region':['max', 'min'], 'charges' : np.ptp}, exclude_value = None)
P.runPAVA(sign = 'auto')

# There are four attributes that will be stored in the PAVA object after running `runPAVA`
print(P.OrgDataAssignment)
print(P.CSD_Summary)
print(P.GCM_Summary)
print(P.PAV_Summary)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/age-on-insurance-PAVA-summary.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> PAVA.PAV_Summary output </sub> </p>

```python
# Visualize the Cumulative Sum Diagram and Greatest Convex Minorant of the CSD
MOB_PLOT.plotPAVACsd(CSD_Summary = P.CSD_Summary)

# apply the PAVA result to a pandas.Series and return either a interval display or a value of starting or closing value of the interval range.
_pavRes = P.applyPAVA(df['age'], 'interval')
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/age-insuranceclaim-PAVA.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> PAVA CSD & GCM visualization example </sub> </p>

<h2> <strong> Highlighted Features </strong></h2>

### **_User Preferences_**:

**_Monotonic Optimal Binning_**

The MOB algorithm offers two user preference settings (**`mergeMethod`** argument):

1. `Size`: This setting allows you to optimize the sample size of each bin within specified maximum and minimum limits while ensuring that the minimum number of bins constraint is maintained.

2. `Stats`: With this setting, the algorithm applies a stricter approach based on hypothesis testing results.<br>


Typically, the `'Stats'` (statistical-based) and `'Size'` (bin size-based) methods yield identical results. However, when dealing with data under certain scenarios where the `'Size'` method, employed by **MOB**, tends to prioritize maintaining the population of each bin within the maximum and minimum limits. In contrast, the `'Stats'` method adheres to a more rigorous logic based on the results of hypothesis testing.

For example, following the previos example code.

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
StatsBinning = MOB_ALGO.runMOB(mergeMethod='Stats')

# plot the bin summary data.
print('Bins Size Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning)
print('Statisitcal Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning)
```

<div>
  <table >
    <thead>
      <tr>
        <th style="text-align: center;">SizeBinning</th>
        <th style="text-align: center;">StatsBinning</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center;">runMOB(mergeMethod='Size') (bins size base)</td>
        <td style="text-align: center;">runMOB(mergeMethod='Stats') (statistical base)</td>
      </tr>
      <tr>
        <td>
            <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/Creditamount-Size.png" width="400" >
        </td>
        <td>
           <img src="https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/Creditamount-Stats.png" width="400">
        </td>
      </tr>
    </tbody>
  </table>
</div>

The left side image is the result generated by **`mergeMethod = 'Size'`** (bin size-based), and the right side is the result generated by **`mergeMethod = 'Stats'`** (statistical-based). We can see that the `'Size'` method is designed to merge bins that fail to meet the minimum sample population requirement. This approach ensures that the number of bins remains within the specified limit, preventing it from exceeding the minimum bin limitation. By merging bins that fall short of the population threshold, the `'Size'` method effectively maintains a balanced distribution of data across the bins..<br><br>

### **_Extended Application_**:

**_Pool Adjacent Violators Algorithm_**

The **[PAVA](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-PAVA-PAVA.md)** module allows users to include additional variables in a dictionary format, along with their corresponding statistics metrics, which will be displayed in the final PAVA result table. Unlike the MOB module, which is limited to calculating the mean (average) for the metric, the PAVA module accepts multiple statistics options, valid inputs include `'count'`, `'mean'`, `'sum'`, `'std'`, `'var'`, `'min'`, `'max'`, `'ptp'`. It's important to note that the `add_var_aggFunc` argument usage is similar to inputting a dictionary for different aggregations per column, referring to **[pandas.DataFrame.agg](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html)**. See [PAVA documentation](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-PAVA-PAVA.md) for more information.

```python
# construct the PAVA object
P = PAVA(data = df, var = 'age', response = 'insuranceclaim', metric='mean', 
         add_var_aggFunc = {'bmi':'mean', 'smoker':'sum', 'region':['max', 'min'], 'charges' : np.ptp}, exclude_value = None)
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/age-on-insurance-PAVA-summary.png' alt = 'Image' style = 'width: 1200px'/></p>

<p align = 'center'> <sub> PAVA.PAV_Summary output </sub> </p>


<h2><strong> Full Documentation </strong></h2>

↪ [Full API Reference](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)<br><br>

<h2> <strong> Environment </strong></h2>

```bash
OS : macOS Ventura

IDE: Visual Studio Code 

Language : Python 3.9.7 
    - pandas 1.3.4
    - numpy 1.20.3
    - scipy 1.7.1
    - matplotlib 3.7.1
    - typing 3.7.4.3
```


<h2><strong> Reference </strong></h2>

- [Mironchyk, Pavel, and Viktor Tchistiakov. "Monotone optimal binning algorithm for credit risk modeling." Utr. Work. Pap (2017).](https://www.researchgate.net/profile/Viktor-Tchistiakov/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/links/5a5dd1a8458515c03edf9a97/Monotone-optimal-binning-algorithm-for-credit-risk-modeling.pdf)
  
- [Smalbil, P. J. "The choices of weights in the iterative convex minorant algorithm." (2015).](https://repository.tudelft.nl/islandora/object/uuid:5a111157-1a92-4176-9c8e-0b848feb7c30)

- Testing Dataset 1 : [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) from [Kaggle](https://www.kaggle.com/)

- Testing Dataset 2 : [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset) from [Kaggle](https://www.kaggle.com/)

- GitHub Project : [Monotone Optimal Binning (SAS 9.4 version)](https://github.com/cdfq384903/MonotonicOptimalBinning)

<h2><strong> Authors </strong></h2>


1. Ta-Hung (Denny) Chen
    - LinkedIn Profile : https://www.linkedin.com/in/dennychen-tahung/
    - E-Mail : denny20700@gmail.com
2. Yu-Cheng (Darren) Tsai
    - LindedIn Profile : https://www.linkedin.com/in/darren-yucheng-tsai/
    - E-Mail : 
3. Peter Chen
   - LinkedIn Profile : https://www.linkedin.com/in/peterchentsungwei/
   - E-Mail : peterwei20700@gmail.com

