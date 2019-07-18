# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys
from collections import defaultdict, OrderedDict
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
# Soothsayer
from ..utils import is_dict

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
    adjusted = multipletests(p_values, method=method, **kwargs)[1]
    if index is not None:
        return pd.Series(adjusted, index=index, name=name)
    else:
        return adjusted






#
