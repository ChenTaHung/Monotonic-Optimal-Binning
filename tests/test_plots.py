import os
import tempfile
import numpy as np
import pandas as pd

from MOBPY.core.pava import PAVA
from MOBPY.plot.csd_gcm import plot_csd_gcm
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints
from MOBPY.plot.mob_plot import MOBPlot


def test_plot_csd_gcm_smoke():
    # small, clean dataset
    x = np.repeat([0.0, 1.0, 2.0], repeats=10)
    y = np.concatenate([np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)])
    df = pd.DataFrame({"x": x, "y": y})
    p = PAVA(df=df, x="x", y="y").fit()

    # should render without error
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "csd_gcm.png")
        plot_csd_gcm(groups_df=p.groups_, blocks=p.export_blocks(as_dict=True), x_name="x", y_name="y", savepath=path)
        assert os.path.exists(path)


def test_mob_plot_bins_summary_smoke_binary():
    rng = np.random.default_rng(7)
    n = 250
    x = np.linspace(-1, 2, n) + rng.normal(0, 0.05, n)
    p = 1 / (1 + np.exp(-1.8 * x))
    y = rng.binomial(1, p, size=n)
    df = pd.DataFrame({"x": x, "y": y})

    b = MonotonicBinner(df=df, x="x", y="y", constraints=BinningConstraints(max_bins=5, min_bins=2)).fit()
    summary = b.summary_()
    # ensure required columns exist for plotting
    for col in ["interval", "nsamples", "bads", "goods", "bad_rate", "woe", "iv_grp"]:
        assert col in summary.columns

    # smoke: render WoE/bad-rate plot
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mob_summary.png")
        MOBPlot.plot_bins_summary(summary, savepath=path)
        assert os.path.exists(path)
