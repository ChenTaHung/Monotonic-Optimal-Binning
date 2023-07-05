<h1><p align = 'center'><strong> Monotonic-Optimal-Binning</strong> </p></h1>

**MOB** is a statistical approach to transform continuous variables into optimal and monotonic categorical variables. In this proejct, we extend the application so that the user can choose whether merge the bins under a `statistics` base or a `bins size` base to obtain the optimal result based on the users' expectation.<br>

<h2><strong> Installation </strong></h2>

```bash
python3 -m pip install MOBPY
```

<h2><strong> Usage </strong></h2>


<h3><span style = 'font-size:larger'> Example :</span></h3>

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

<p align = 'center'><img src = 'https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/images/Durationinmonth%20bins%20summary.png' alt = 'Image' style = 'width: 1200px'/></p>

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




Normally, the result of `Stats` (statistical base) and `Size` (bins size base) will be identical, but when the data appears to be quite extreme in the binning process, the `Size` method will prefer to make the population of each bin between the maximum and minimum limitation, while the `Stats` method will remain to conduct the algorithm through a stricter logic based on the testing hypothesis results.

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

The left side image is the result generated by **`mergeMethod = 'Size'`** (bins size base), and the right side is the result generated by **`mergeMethod = 'Stats'`** (statistical base).We can see that the **Size method** merge the bins that do not meet the minimum sample population and maintain the bins number in order to prevent from exceeding the minimum bins limitation.<br><br>



<h2> <strong> Environment </strong></h2>

```bash
OS : macOS Ventura

IDE: Visual Studio Code 1.79.2 (Universal)

Language : Python 3.9.7 
    - pandas 1.3.4
    - numpy 1.20.3
    - scipy 1.7.1
    - matplotlib 3.7.1
```
<br>

<h2><strong> Citation </strong></h2>

- [Mironchyk, Pavel, and Viktor Tchistiakov. "Monotone optimal binning algorithm for credit risk modeling." Utr. Work. Pap (2017).](https://www.researchgate.net/profile/Viktor-Tchistiakov/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/links/5a5dd1a8458515c03edf9a97/Monotone-optimal-binning-algorithm-for-credit-risk-modeling.pdf)



<h2><strong> Reference </strong></h2>

- Testing Dataset : [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) from [Kaggle](https://www.kaggle.com/)

- GitHub Project : [Monotone Optimal Binning (SAS 9.4 version)](https://github.com/cdfq384903/MonotonicOptimalBinning)

<h2><strong> Authors </strong></h2>


1. Chen, Ta-Hung (Denny) <br>
    - LinkedIn Profile : https://www.linkedin.com/in/dennychen-tahung/
    - E-Mail : denny20700@gmail.com
2. Tsai, Yu-Cheng (Darren)
    - LindedIn Profile : https://www.linkedin.com/in/darren-yucheng-tsai/
    - E-Mail : 


