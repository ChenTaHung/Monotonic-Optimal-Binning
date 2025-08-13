"""Plotting utilities for MOBPY.

Import submodules directly to avoid eager imports during package init, e.g.:

    from MOBPY.plot.csd_gcm import plot_csd_gcm, plot_csd_gcm_from_binner
    from MOBPY.plot.mob_plot import MOBPlot

This keeps the package import lightweight and prevents CI/import-order issues
when only a subset of plotting functionality is needed.
"""

# Do not eagerly import submodules here.
# This avoids failures when only one submodule is needed (e.g., csd_gcm)
# and the other may have optional dependencies or is not present.
__all__: list[str] = []
