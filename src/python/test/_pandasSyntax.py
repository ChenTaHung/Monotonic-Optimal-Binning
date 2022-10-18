#%%
import pandas as pd
import numpy as np
#%%
df = pd.read_csv("/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv")
# %%
df["Durationinmonth"].dtypes
# %%
df["default"] = df["default"]-1
#%%
df1 = df.groupby("Durationinmonth")["default"].agg(["count", "mean", "sum"]).sort_index().reset_index()
# %%
df1.insert(1, "end", df1["Durationinmonth"])
df1
# %%

# %%
