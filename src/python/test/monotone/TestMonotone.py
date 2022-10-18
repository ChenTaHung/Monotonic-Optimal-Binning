import pandas as pd
import numpy as np
import os
os.chdir("/Users/chentahung/Desktop/git/mob-py/src/python/main")
from monotone.MonotoneTuner import monotoneTuner

def main() :
    df = pd.read_csv("/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv")
    df["default"] = df["default"] - 1
    Tuner = monotoneTuner(data = df, var = "Durationinmonth", response = "default", metric = "mean")
    out = Tuner.tuneMonotone(monoTarget = "mean")
    return out


if __name__ == '__main__' :
    res = main()
    print(res)