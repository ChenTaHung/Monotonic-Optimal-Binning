<h1><strong><p align = center> MOBPY Documentation Index </p></strong></h1>

<h2><p  align=center style = 'font-style:italic'><strong>API Reference</p></strong></p></h2>


```
   src
    ├── MOBPY
    │   ├── MOB.py
    │   ├── __ init __.py
    │   ├── categorical
    │   │   └── __init__.py
    │   ├── numeric
    │   │   ├── Monotone.py
    │   │   ├── MonotoneNode.py
    │   │   ├── OptimalBinning.py
    │   │   └── __init__.py
    │   └── plot
    │       ├── MOB_PLOT.py
    │       └── __init__.py
```

## **_MOBPY.MOB_**

**[MOB](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-MOB-MOB.md)()** : constructor to construct the `MOB` object.

**MOB.[setBinningConstraints](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-MOB-MOB-setBinningConstraints.md)()** :  Set the binning constraints. A must-do step before execute the **`runMOB`**().

**MOB.[runMOB](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-MOB-MOB-runMOB.md)()** : Execute the program to conduct MOB alogorithm to discretize the gieven variable `var`(See [MOB](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-MOB-MOB.md)).

## **_MOBPY.plot_** 

**[MOB_PLOT](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-plot-MOB_PLOT-MOB_PLOT.md)** : Module to call the static method **`plotBinsSummary`**().

**MOB_PLOT.[plotBinsSummary](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-plot-MOB_PLOT-MOB_PLOT-plotBinsSummary.md)()** : Visualize the binning summary generated by **`MOB.runMOB`**().

<br><br>

`↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓`

[Jump Back to `MOBPY Documentation Index`](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/blob/main/doc/MOBPY-API-Ref.md)