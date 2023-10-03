# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, warnings
from collections import OrderedDict, defaultdict
from collections.abc import Mapping 

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
# Soothsayer
from ..utils import is_dict, assert_acceptable_arguments, flatten


__all__ = ["shannon_entropy","kullback_leibler_divergence", "welchs_ttest", "p_adjust", "statistically_significant_symbols", "biweight_midcorrelation", "differential_abundance", "feature_set_enrichment"]

# Shannon entropy
def shannon_entropy(pk):
    """
    Simple wrapper around scipy.stats.entropy with base set to 2
    """
    return stats.entropy(pk, base=2)

# Kullback Leibler Divergence
def kullback_leibler_divergence(pk, qk, base=2):
    """
    Simple wrapper around scipy.stats.entropy with base set to 2
    """
    return stats.entropy(pk=pk, qk=qk, base=base)

# Welch's T-Test
def welchs_ttest(a,b,nan_policy="propogate"):
    """
    Simple wrapper around scipy.stats.ttest_ind
    """
    return stats.ttest_ind(a=a, b=b, equal_var=False, nan_policy=nan_policy)

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

# Differential abundance
def differential_abundance(X:pd.DataFrame, y:pd.Series, reference_class=None, design_matrix=None, method="ALDEx2", into=pd.DataFrame, algo_kws=dict(), random_state=0, show_console=False, **kwargs):
    # Assertions (This is duplicate but would rather be safe)
    assert np.all(X.shape[0] == y.size), "X.shape[0] != y.size"
    assert np.all(X.index == y.index), "X.index != y.index"

    #Methods
    assert_acceptable_arguments(method, {"ALDEx2", "edgeR::exactTest", "edgeR::glmLRT"})

    # ALDEx2
    if method == "ALDEx2":
        from soothsayer.r_wrappers import ALDEx2
        if design_matrix is not None:
            warnings.warn("`design_matrix` not used for method='ALDEx2'")
        kwargs = {
            "X":X,
            "y":y,
            "reference_class":reference_class,
            "into":into,
            "aldex2_kws":algo_kws,
            "random_state":random_state,
            "show_console":show_console,
            **kwargs,
        }
        return ALDEx2.run_aldex2(**kwargs)
    
    # EdgeR's ExactTest
    if method == "edgeR::exactTest":
        if design_matrix is not None:
            warnings.warn("`design_matrix` not used for method='edgeR::exactTest'")
        from soothsayer.r_wrappers import edgeR
        kwargs = {
            "X":X,
            "y":y,
            "reference_class":reference_class,
            "into":into,
            "edger_kws":algo_kws,
            "random_state":random_state,
            "show_console":show_console,
            **kwargs,
        }
        return edgeR.run_edger_exact_test(**kwargs)
    
    # EdgeR's GLM
    if method == "edgeR::glmLRT":
        from soothsayer.r_wrappers import edgeR
        kwargs = {
            "X":X,
            "y":y,
            "design_matrix":design_matrix,
            "reference_class":reference_class,
            "into":into,
            "edger_kws":algo_kws,
            "random_state":random_state,
            "show_console":show_console,
            **kwargs,
        }
        return edgeR.run_edger_glm(**kwargs)


# Feature Set Enrichment
def feature_set_enrichment(features:set, feature_sets:Mapping,  tol_test:float=0.05, test_method:str="hypergeometric", fdr_method:str="fdr_bh"): # How to incorporate weights?
    """
    Future: 
     * Incorporate `feature_weights:pd.Series` using Wallenius' noncentral hypergeometric distribution

    Theory: 
    http://pedagogix-tagc.univ-mrs.fr/courses/ASG1/practicals/go_statistics_td/go_statistics_td_2015.html
    """
    assert_acceptable_arguments(test_method, {"hypergeometric"})
    
    # Force set type for features in case other iterables are used
    features = set(features)

    # Union of all features in feature sets
    feature_set_union = flatten(feature_sets, into=set)
    number_of_features_in_db = len(feature_set_union)
    
    data = defaultdict(OrderedDict)
    
    for name, feature_set in feature_sets.items():
        # Force feature_set as set type
        feature_set = set(feature_set)
        # Get number of features in set
        number_of_features_in_set = len(feature_set)
        # Get number of features in query
        number_of_features_in_query = len(features)
        # Get number of overlapping features between query and set
        number_of_features_overlapping = len(features & feature_set)
        
        # Rum hypergeometric test
        if test_method == "hypergeometric":
            model = stats.hypergeom(
                M = number_of_features_in_db,
                n = number_of_features_in_set,
                N = number_of_features_in_query,
            )
            # "We want the *inclusive* upper tail : P-value = P(X≥x). 
            # For this, we can compute the exclusive upper tail of the value just below x. 
            # Indeed, since the distribution is discrete, P(X >x-1) = P(X ≥x)."
            # Source - http://pedagogix-tagc.univ-mrs.fr/courses/ASG1/practicals/go_statistics_td/go_statistics_td_2015.html
            p_value = model.sf(number_of_features_overlapping - 1)
            data["p_value"][name] = p_value

#         if test_method == "fisher":
#             number_of_query_not_in_set = len(features - feature_set)
#             number_of_features_not_in_set = number_of_features_in_db - number_of_features_in_set
#             contingency_table = [
#                 [number_of_features_overlapping, number_of_features_in_query - number_of_features_overlapping],
#                 [number_of_features_in_set - number_of_features_overlapping, number_of_features_not_in_set - (number_of_features_in_query - number_of_features_overlapping)],
#             ]
#             stat, p_value = stats.fisher_exact(contingency_table, alternative="greater")
#             data["contingency_table"][name] = contingency_table
#             data["stat"][name] = stat
#             data["p_value"][name] = p_value

        # Store values
        data["number_of_features_in_db (M)"][name] = number_of_features_in_db
        data["number_of_features_in_set (n)"][name] = number_of_features_in_set
        data["number_of_features_in_query (N)"][name] = number_of_features_in_query
        data["number_of_features_overlapping (k)"][name] = number_of_features_overlapping

    # Create dataframe
    df = pd.DataFrame(data)
    df.insert(0, "test_method", test_method)

    # Calculate adjusted p-value
    df.insert(df.shape[1], "fdr_method", fdr_method)
    df.insert(df.shape[1], "fdr_value", p_adjust(df["p_value"], method=fdr_method))
    
    # Determine statistical significance
    if tol_test is not None:
        df.insert(df.shape[1], "fdr<{}".format(tol_test), df["fdr_value"] < tol_test)
    return df

