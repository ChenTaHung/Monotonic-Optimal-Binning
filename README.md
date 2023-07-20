<h1><p align = 'center'><strong> Monotonic-Optimal-Binning </strong> </p></h1>
<h3><p align = 'center'><strong> Python implementation (MOBPY) </strong> </p></h3>

[**MOB**](https://pypi.org/project/MOBPY/) is a statistical approach to transform continuous variables into optimal and monotonic categorical variables. In this project, we have expanded the application to allow the users to merge the bins based on `statistics` or `bin size`. This is a Python-based project that enables the users to achieve monotone optimal binning results aligned with their expectations.<br>

<h2><strong> Installation </strong></h2>

```bash
python3 -m pip install MOBPY
```

<h2><strong> Usage </strong></h2>


**_Example_**:

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
    # A must-do step is to set the binning constraints.
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

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/Durationinmonth%20bins%20summary.png' alt = 'Image' style = 'width: 800px'/></p>

And after we receive the binning result dataframe, we can plot it by using `MOBPY.plot.MOB_PLOT.plotBinsSummary` to visualize the binning summary result.

```python
from MOBPY.plot.MOB_PLOT import MOB_PLOT

# plot the bin summary data.
print('Bins Size Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')

print('Statisitcal Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = 'Durationinmonth')
```

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/charts/Durationinmonth-Size.png' alt = 'Image' style = 'width: 1200px'/></p>


<h2> <strong> Highlighted Features </strong></h2>

**_User Preferences_**:

The MOB algorithm offers two user preference settings (**`mergeMethod`** argument):

1. `Size`: This setting allows you to optimize the sample size of each bin within specified maximum and minimum limits while ensuring that the minimum number of bins constraint is maintained.

2. `Stats`: With this setting, the algorithm applies a stricter approach based on hypothesis testing results.<br>


Typically, the `'Stats'` (statistical-based) and `'Size'` (bin size-based) methods yield identical results. However, when dealing with data under certain scenarios where the `'Size'` method, employed by **MOB**, tends to prioritize maintaining the population of each bin within the maximum and minimum limits. In contrast, the `'Stats'` method adheres to a more rigorous logic based on the results of hypothesis testing.

For example, 

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
MOB_PLOT.plotBinsSummary(monoOptBinTable = SizeBinning, var_name = 'Durationinmonth')
print('Statisitcal Base')
MOB_PLOT.plotBinsSummary(monoOptBinTable = StatsBinning, var_name = 'Durationinmonth')
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


<h2><strong> Full Documentation </strong></h2>

↪ [Full API Reference](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)<br><br>

<h2> <strong> Environment </strong></h2>

```bash
OS : macOS Ventura

IDE: Visual Studio Code 1.79.2 (Universal)

Language : Python 3.9.7 
    - pandas 1.3.4
    - numpy 1.20.3
    - scipy 1.7.1
    - matplotlib 3.7.1
    - typing==3.7.4.3
```


<h2><strong> Reference </strong></h2>

- Testing Dataset : [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) from [Kaggle](https://www.kaggle.com/)

- [Mironchyk, Pavel, and Viktor Tchistiakov. "Monotone optimal binning algorithm for credit risk modeling." Utr. Work. Pap (2017).](https://www.researchgate.net/profile/Viktor-Tchistiakov/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/links/5a5dd1a8458515c03edf9a97/Monotone-optimal-binning-algorithm-for-credit-risk-modeling.pdf)

- GitHub Project : [Monotone Optimal Binning (SAS 9.4 version)](https://github.com/cdfq384903/MonotonicOptimalBinning)

<h2><strong> Authors </strong></h2>


1. Chen, Ta-Hung (Denny) <br>
    - LinkedIn Profile : https://www.linkedin.com/in/dennychen-tahung/
    - E-Mail : denny20700@gmail.com
2. Tsai, Yu-Cheng (Darren)
    - LindedIn Profile : https://www.linkedin.com/in/darren-yucheng-tsai/
    - E-Mail : 


