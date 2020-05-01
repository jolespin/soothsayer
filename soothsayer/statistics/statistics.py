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
from ..utils import is_dict, assert_acceptable_arguments

__all__ = ["p_adjust", "statistically_significant_symbols", "biweight_midcorrelation"]

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

# Biweight midcorrelation
def biweight_midcorrelation(a,b, check_index_order=True, use_numba=False, verbose=False):
    """
    a,b: {np.array, pd.Series}

    Code adapted from the following sources:
        * https://stackoverflow.com/questions/61090539/how-can-i-use-broadcasting-with-numpy-to-speed-up-this-correlation-calculation/61219867#61219867
        * https://github.com/olgabot/pandas/blob/e8caf4c09e1a505eb3c88b475bc44d9389956585/pandas/core/nanops.py

    Special thanks to the following people:
        * @norok2 (https://stackoverflow.com/users/5218354/norok2) for optimization (vectorization and numba)
        * @olgabot (https://github.com/olgabot) for NumPy implementation

    Benchmarking:
        * iris_features (4)
            * numba: 321 ms ± 15.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            * numpy: 478 µs ± 8.37 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        * iris_samples: (150)
            * numba: 312 ms ± 8.54 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            * numpy: 438 µs ± 5.57 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Future:
        * Handle missing values

    """
    # Data
    result = None
    labels = None
    assert type(a) is type(b), "a & b must both either be np.array or pd.Series"
    if isinstance(a, pd.Series):
        if check_index_order:
            assert np.all(a.index == b.index), "a.index and b.index must be be the same ordering"
        labels = a.index
        a = a.values
        b = b.values

    # Base computation
    def _base_computation(a,b):
        n = a.size
        a = a - np.median(a)
        b = b - np.median(b)
        v_a = 1 - (a / (9 * np.median(np.abs(a)))) ** 2
        v_b = 1 - (b / (9 * np.median(np.abs(b)))) ** 2
        a = (v_a > 0) * a * v_a ** 2
        b = (v_b > 0) * b * v_b ** 2
        return n, a, b

    # Check if numba is available
    assert_acceptable_arguments(use_numba, {True, False, "infer"})
    if use_numba == "infer":
        if "numba" in sys.modules:
            use_numba = True
        else:
            use_numba = False
        if verbose:
            print("Numba is available:", use_numba, file=sys.stderr)

    # Compute using numba
    if use_numba:
        assert "numba" in sys.modules
        from numba import jit

        @jit
        def _biweight_midcorrelation_numba(a, b):
            n, a, b = _base_computation(a,b)
            s_ab = s_aa = s_bb = 0
            for i in range(n):
                s_ab += a[i] * b[i]
                s_aa += a[i] * a[i]
                s_bb += b[i] * b[i]
            return s_ab / np.sqrt(s_aa) / np.sqrt(s_bb)

        result = _biweight_midcorrelation_numba(a,b)

    # Compute using numpy
    else:
        def _biweight_midcorrelation_numpy(a, b):
            n, a, b = _base_computation(a,b)
            return np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))
        result = _biweight_midcorrelation_numpy(a,b)

    # Add labels
    if labels is not None:
        result = pd.Series(result, index=labels)

    return result
