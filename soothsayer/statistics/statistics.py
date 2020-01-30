# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
# Soothsayer
from ..utils import is_dict

__all__ = ["p_adjust", "statistically_significant_symbols"]

# Adjust p-values
def p_adjust(p_values:pd.Series, method="fdr", name=None, **kwargs):
    """
    Multiple test correction to adjust p-values

    fdr -> fdr_bh # https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html

    `bonferroni` : one-step correction
    `sidak` : one-step correction
    `holm-sidak` : step down method using Sidak adjustments
    `holm` : step-down method using Bonferroni adjustments
    `simes-hochberg` : step-up method  (independent)
    `hommel` : closed method based on Simes tests (non-negative)
    `fdr_bh` : Benjamini/Hochberg  (non-negative)
    `fdr_by` : Benjamini/Yekutieli (negative)
    `fdr_tsbh` : two stage fdr correction (non-negative)
    `fdr_tsbky` : two stage fdr correction (non-negative)

    """
    # Check type of input data
    index = None
    if is_dict(p_values):
        p_values = pd.Series(p_values)
    if isinstance(p_values, pd.Series):
        index = p_values.index
        p_values = p_values.values
    if method == "fdr":
        method = "fdr_bh"

    if index is not None:
        mask_null = np.isnan(p_values)
        if np.any(mask_null):
            scaffold = np.ones_like(p_values)*np.nan
            scaffold[np.logical_not(mask_null)] = multipletests(p_values[np.logical_not(mask_null)], method=method, **kwargs)[1]
            return pd.Series(scaffold, index=index)
        else:
            adjusted = multipletests(p_values, method=method, **kwargs)[1]
            return pd.Series(adjusted, index=index, name=name)
    else:
        num_nan = np.isnan(p_values).sum()
        assert  num_nan == 0, "Please remove the {} missing values".format(num_nan)
        return multipletests(p_values, method=method, **kwargs)[1]

# Statistically significant symbols
def statistically_significant_symbols(p:pd.Series, not_significant="ns", return_pvalues=False):
    """
    # Lexicon
    # ns
    # P > 0.05
    # *
    # P ≤ 0.05
    # **
    # P ≤ 0.01
    # ***
    # P ≤ 0.001
    # ****
    #  P ≤ 0.0001 (For the last two choices only)

    Future:  Make this customizable
    """
    if not hasattr(p, "__iter__"):
        if p > 0.05:
            return not_significant
        symbol = ""
        if p <= 0.05:
            symbol += "*"
        if p <= 0.01:
            symbol += "*"
        if p <= 0.001:
            symbol += "*"
        if p <= 0.0001:
            symbol += "*"
        return symbol
    else:
        symbols =  pd.Series(p).map(lambda x:statistically_significant_symbols(x, not_significant=not_significant))
        if return_pvalues:
            return pd.concat([symbols.to_frame("symbol"), p.to_frame("p_value")], axis=1)
        else:
            return symbols



#
