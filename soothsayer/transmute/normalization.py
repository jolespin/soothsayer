# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, warnings
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np

# Compositional
# from skbio.stats.composition import clr, ilr
# from gneiss.composition import ilr_transform
import ete3
import compositional as coda

# SciPy
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Soothsayer
from .conversion import ete_to_skbio
from ..r_wrappers.packages.edgeR import normalize_edgeR
from ..r_wrappers.packages.metagenomeSeq import normalize_css
# from soothsayer.r_wrappers.packages.philr import normalize_philr


from ..utils import is_dict, assert_acceptable_arguments, is_number

__all__ = {"normalize_minmax", "normalize_tss", "normalize_center", "normalize_zscore", "normalize_quantile", "normalize_boxcox",  "normalize", "normalize_expression"}

functions_from_compositional = {"transform_clr","transform_xlr", "transform_iqlr", "transform_ilr"}

for function_name in functions_from_compositional:
    globals()[function_name] = getattr(coda, function_name)
    __all__.add(function_name)

__all__ = sorted(__all__)
   

# ==============================================================================
# Normalization
# ==============================================================================
# Minmax
def normalize_minmax(X, feature_range=(0,1)):
    assert isinstance(X, (pd.Series, pd.DataFrame)), "X must be a pd.Series or pd.DataFrame"
    if isinstance(X, pd.Series):
        values = MinMaxScaler(feature_range=feature_range).fit_transform(X.values.reshape(-1,1)).ravel()
        return pd.Series(values, index=X.index, name=X.name)
    if isinstance(X, pd.DataFrame):
        return X.apply(lambda x: normalize_minmax(x, feature_range=feature_range), axis=1)

# Normalize by summing to one
def normalize_tss(X):
    """
    NumPy or Pandas
    """
    if len(X.shape) == 2: # if type(X) is pd.DataFrame:
        sum_ = X.sum(axis=1)
        return (X.T/sum_).T
    if len(X.shape) == 1: #elif type(X) is pd.Series:
        return X/X.sum()

# Center Normalization
def normalize_center(X):
    if isinstance(X, pd.DataFrame):
        A = X.values
        return pd.DataFrame(A - np.nanmean(X, axis=1).reshape((-1,1)), index=X.index, columns=X.columns)
    elif isinstance(X, pd.Series):
        return X - X.mean()

# Zscore Normalization
def normalize_zscore(X):
    if isinstance(X, pd.DataFrame):
        A_center = normalize_center(X).values
        A_std = np.nanstd(X.values, axis=1).reshape((-1,1))
        return pd.DataFrame(A_center/A_std, index=X.index, columns=X.columns)
    elif isinstance(X, pd.Series):
        return (X - X.mean())/X.std()

# Quantile Normalization
def normalize_quantile(X):
    assert isinstance(X, pd.DataFrame), "Quantile normalization is only compatible with `pd.DataFrame`"
    X_transpose = X.T
    X_mu = X_transpose.stack().groupby(X_transpose.rank(method='first').stack().astype(int)).mean()
    return X_transpose.rank(method='min').stack().astype(int).map(X_mu).unstack().T

# Boxcox Normalization
def normalize_boxcox(X):
    if isinstance(X, pd.DataFrame):
        return X.apply(lambda x:stats.boxcox(x)[0], axis=1)
    elif isinstance(X, pd.Series):
        return pd.Series(stats.boxcox(X)[0],index=X.index, name=X.name)

# Normalization
def normalize(X, method="tss", axis=1, tree=None, feature_range=(0,1)):
    """
    Assumes X_data.index = Samples, X_data.columns = Attributes
    axis = 0 = cols, axis = 1 = rows
    e.g. axis=1, method=ratio: Normalize for relative abundance so each row sums to 1.
    "tss" = total-sum-scaling
    "center" = mean centered
    "clr" = center log-ratio
    "quantile" = quantile normalization
    "zscore" = zscore normalization (mu: 0, var:1)
    "boxcox"
    "ilr" = isometric log ratio transform
    """
    if method in ["ratio", "relative-abundance"]:
        method = "tss"
    # Transpose for axis == 0
    if axis == 0:
        X = X.T
    # Common
    if method == "tss":
        df_normalized = normalize_tss(X)
    if method == "center":
        df_normalized = normalize_center(X)
    if method == "zscore":
        df_normalized = normalize_zscore(X)
    if method == "quantile":
        df_normalized = normalize_quantile(X)
    if method == "boxcox":
        df_normalized = normalize_boxcox(X)
    if method == "minmax":
        df_normalized = normalize_minmax(X, feature_range=feature_range)
    # Aitchison
    if method == "clr":
        df_normalized = transform_clr(X)
    if method == "ilr":
        df_normalized = transform_ilr(X, tree=tree)
    # if method == "philr":
    #     df_normalized = normalize_philr(X, tree=tree)

    # Transpose back
    if axis == 0:
        df_normalized = df_normalized.T
    return df_normalized

# Normalize gene expresssion data
def normalize_expression(X:pd.DataFrame, method:str="tpm", length:pd.Series=None, p=0.75, kws=dict()):
    """
    # FPKM
    Fragments Per Kilobase of transcript per Million mapped reads
        C = Number of reads mapped to a gene
        N = Total mapped reads in the experiment
        L = exon length in base-pairs for a gene
        Equation: RPKM = (10^9 * C)/(N * L)
            or
        numReads / ( geneLength/1000 * totalNumReads/1,000,000 ) # C/((L/1e3).T*(N/1e6)).T

    # TPM
    Transcripts Per Kilobase Million
        (1) Divide the read counts by the length of each gene in kilobases. This gives you reads per kilobase (RPK).
        (2) Count up all the RPK values in a sample and divide this number by 1,000,000. This is your “per million” scaling factor.
        (3) Divide the RPK values by the “per million” scaling factor. This gives you TPM.

    # TMM
    https://genomebiology.biomedcentral.com/articles/10.1186/gb-2010-11-3-r25

    # GeTMM
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2246-7#MOESM4

    # RLE
    https://genomebiology.biomedcentral.com/articles/10.1186/gb-2010-11-10-r106

    # Upperquartile
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-94

    # CSS
    https://github.com/biocore/qiime/blob/6d8ffca3ffe2bb40057b8f9cbc08e2029b96d790/qiime/support_files/R/CSS.r

    # Notes:
    Proper pseudocount addition would be the following as shown by metagenomeSeq's MRcounts
    log(X_normalized + 1)
    """
    if method is None:
        return X

    method = method.lower()
    assert_acceptable_arguments(query=[method], target={"rpk", "fpkm", "rpkm", "tpm", "tmm", "getmm", "rle", "upperquartile", "css"})


    # Lengths
    if method in {"fpkm", "rpkm", "rpk",  "tpm", "getmm"}:
        assert length is not None, "If FPKM, RPKM, TPM, or GeTMM is chosed as the method then `length` cannot be None.  It must be either a pd.Series of sequences or sequence lengths"
        length = pd.Series(length)[X.columns]
        assert length.isnull().sum() == 0, "Not all of the genes in `X.columns` are in `length.index`.  Either use a different normalization or get the missing sequence lengths"
        # If sequences are given then convert to length (assumes CDS and no introns)
        if pd.api.types.is_string_dtype(length):
            length = length.map(len)

    # FPKM, RPKM, and TPM normalization
    if method in {"fpkm", "rpkm", "rpk", "tpm"}:

        # Set up variables
        C = X.values
        L = length.values
        N = X.sum(axis=1).values.reshape(-1,1)


        if method in {"fpkm","rpkm"}:
            # Compute operations
            numerator = 1e9 * C
            denominator = (N*L)
            return pd.DataFrame(numerator/denominator, index=X.index, columns=X.columns)

        if method in {"rpk", "tpm"}:
            rpk = C/L
            if method == "rpk":
                return pd.DataFrame(rpk, index=X.index, columns=X.columns)
            if method == "tpm":
                per_million_scaling_factor = (rpk.sum(axis=1)/1e6).reshape(-1,1)
                return pd.DataFrame( rpk/per_million_scaling_factor, index=X.index, columns=X.columns)


    # TMM/GeTMM Normalization
    if method in {"tmm", "getmm", "rle", "upperquartile"}:
        return normalize_edgeR(X, length=length, method=method, p=p, **kws)
    if method in  {"css"}:
        return normalize_css(X)

# # CLR Normalization
# def transform_clr(X, return_zeros_as_neginfinity=False):
#     """
#     Extension of CLR from skbio to handle zeros and NaN; though, this implementation will be slightly slower.

#     Please refer to the following for CLR documention:
#     http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.clr.html#skbio.stats.composition.clr

#     """
#     assert isinstance(X, (pd.Series, pd.DataFrame)), "Type must be either pd.Series or pd.DataFrame"
#     # X
#     if isinstance(X, pd.DataFrame):

#         return transform_xlr(X, centroid="log_mean",return_zeros_as_neginfinity=return_zeros_as_neginfinity )

#     # x
#     if isinstance(X, pd.Series):
#         # Get values
#         X_values = X.values
#         # Check for zeros
#         X_contains_zeros = False
#         num_zeros = np.any(X == 0).flatten().sum()
#         if num_zeros:
#             warnings.warn("N={} detected in `X`.  Masking them as NaN to perform nan-robust functions".format(num_zeros))
#             X_zero_mask = X == 0
#             X_values[X_zero_mask] = np.nan
#             X_contains_zeros = True

#         # Transform
#         X_log = np.log(X)
#         centroid = np.nanmean(X_log, axis=-1)
#         X_transformed = X_log - centroid

#         # Output
#         if all([return_zeros_as_neginfinity, X_contains_zeros]):
#             X_transformed[X_zero_mask] = -np.inf
#         return pd.Series(X_transformed,index=X.index, name=X.name)

# # ===========================
# # Compositional data analysis
# # ===========================
# # Extension of CLR to use different centroid, references, and zeros without pseudocounts
# def transform_xlr(X:pd.DataFrame, reference_components=None, centroid="log_mean", return_zeros_as_neginfinity=False):
#     """
#     Extension of CLR to incorporate custom metrics such as median and harmonic mean.
#     If you want CLR, please use skbio's implementation as it is faster.
#     This implementation is more versatile with more checks but that makes it slower if it done iteratively.

#     This was designed to handle zero values.  It computes the centroid metrics for all non-zero values (masked as nan)
#     and returns them either as nan (can be used with some correlation functions) or as -inf (for mathematical consistency)

#     Documentation on CLR:
#     http://scikit-bio.org/docs/latest/generated/skbio.stats.composition.clr.html#skbio.stats.composition.clr

#     centroid: {log_median, log_mean, log_hmean, Type[Int]} If int, then that value is used as a percentile (e.g. median is 50)

#     Note: log_mean == arithmetic mean of logs == log of geometric mean
#     """

#     assert np.all(X >= 0), "`X` cannot contain negative values because of log-transformation step."
#     assert len(X.shape) == 2, "`X` must be 2-Dimensional"

#     # Check for labels
#     index = None
#     columns = None
#     X_values = X.astype(float)
#     X_is_labeled = False
#     if isinstance(X, pd.DataFrame):
#         index = X.index
#         columns = X.columns
#         X_values = X.values.astype(float)
#         if reference_components is None:
#             reference_components = columns
#         reference_components = list(map(lambda component: columns.get_loc(component), reference_components))
#         X_is_labeled = True
        
#     # Check centroid
#     if X_is_labeled:
#         if not isinstance(centroid, str):
#             if is_dict(centroid):
#                 centroid = pd.Series(centroid)
#             assert isinstance(centroid, pd.Series), "If `centroid` is dict-like/pd.Series then `X` must be a `pd.DataFrame`."
#             assert set(centroid.index) >= set(index), "Not all indicies from `centroid` are available in `X.index`."
#             centroid = centroid[index].values

#     # Check for zeros
#     X_contains_zeros = False
#     num_zeros = np.any(X == 0).flatten().sum()
#     if num_zeros:
#         warnings.warn("N={} detected in `X`.  Masking them as NaN to perform nan-robust functions".format(num_zeros))
#         X_zero_mask = X == 0
#         X_values[X_zero_mask] = np.nan
#         X_contains_zeros = True

#     # Log transformation
#     X_log = np.log(X_values)

#     # Centroid precomputed
#     if isinstance(centroid, str) or is_number(centroid):

#         # Centroid reference
#         median = {"log_median"}
#         geometric_mean = {"log_mean", "gmean_log" }
#         harmonic_mean = {"log_hmean"}

#         if isinstance(centroid, str):
#             centroid = centroid.lower()
#             assert_acceptable_arguments(centroid, median  | geometric_mean | harmonic_mean)

#             if centroid in median:
#                 centroid = np.nanmedian(X_log[:,reference_components], axis=-1)
#             elif centroid in geometric_mean:
#                 centroid = np.nanmean(X_log[:,reference_components], axis=-1)
#             elif centroid in harmonic_mean:
#                 if X_contains_zeros:
#                     centroid = np.asarray(list(map(lambda x:stats.hmean(x[np.isfinite(x)]), X_log[:,reference_components])))
#                 else:
#                     centroid = stats.hmean(X_log[:,reference_components], axis=-1)

#         # Percentiles
#         if is_number(centroid):
#             centroid = np.asarray(list(map(lambda x:np.percentile(x, centroid), X_log[:,reference_components])))
#     else:
#         assert any([is_dict_like(centroid), np.ndarray]), "If centroid is not a string or a numeric value then it must be either a dict-like/pd.Series (labeled) or numpy array (unlabeled)"

#     # Check dimensions
#     assert len(centroid) == X_log.shape[0], "Dimensionality is not compatible: centroid.size != X.shape[0]."
#     centroid = np.asarray(centroid).reshape(-1,1)
#     # Transform
#     X_transformed = X_log - centroid

#     # Output
#     if all([return_zeros_as_neginfinity, X_contains_zeros]):
#         X_transformed[X_zero_mask] = -np.inf

#     if X_is_labeled:
#         return pd.DataFrame(X_transformed, index=index, columns=columns)
#     else:
#         return X_transformed

# # Interquartile range log-ratio transform
# def transform_iqlr(X:pd.DataFrame, percentile_range=(25,75), centroid="log_mean", interval_type="open", return_zeros_as_neginfinity=False):
#     """
#     Interquartile range log-ratio transform
#     interval: open = (a,b) and closed = [a,b].  open is used by `propr` R package:
#     https://github.com/tpq/propr/blob/2bd7c44bf59eaac6b4d329d38afd40ac83e2089a/R/2-proprCall.R#L31
#     """
#     assert_acceptable_arguments(query=[interval_type],target=["closed", "open"], operation="le")
#     percentile_range = tuple(sorted(percentile_range))
#     assert len(percentile_range) == 2, "percentile_range must have 2 elements"
#     assert isinstance(X, pd.DataFrame), "`transmute_iqlr` can currently only support pd.DataFrames and not np.arrays"
#     # Compute the variance of the CLR transform
#     clr_var = transform_xlr(X, centroid=centroid).var(axis=0)
#     # Remove non-finite values
#     clr_var = clr_var[np.isfinite(clr_var)]
#     # Calculate upper and lower bounds from percentiles
#     lower_bound, upper_bound = np.percentile(clr_var, percentile_range)
#     # Get the reference components
#     if interval_type == "open":
#         reference_components = clr_var[(lower_bound < clr_var) & (clr_var < upper_bound)].index
#     if interval_type == "closed":
#         reference_components = clr_var[(lower_bound <= clr_var) & (clr_var <= upper_bound)].index
#     return transform_xlr(X, reference_components=reference_components, centroid=centroid, return_zeros_as_neginfinity=return_zeros_as_neginfinity)

# # ILR Transformation
# def transform_ilr(X:pd.DataFrame, tree=None, node_prefix="y", bifurcation_kws=dict(), verbose=True):
#     """
#     if `tree` is None then orthonormal basis for Aitchison simplex defaults to J.J.Egozcue orthonormal basis.
#     """
#     assert isinstance(X, pd.DataFrame), "type(X) must be pd.DataFrame"
#     assert not np.any(X == 0), "`X` cannot contain any zero values because of the log-transform.  Give it a pseudocount. (X+1)"

#     # Supply tree
#     if tree is not None:
#         tree_set =  set(tree.get_leaf_names())
#         leaf_set_from_X = set(X.columns)
#         assert leaf_set_from_X <= tree_set, "`X` columns should be a subset of `tree` leaves"
#         tree = tree.copy(method="deepcopy")
#         if leaf_set_from_X < tree_set:
#             n_before_pruning = len(tree_set)
#             tree.prune(leaf_set_from_X)
#             if verbose:
#                 print(f"Pruned {n_before_pruning - len(tree.get_leaves())} attributes to match X.columns", file=sys.stderr)

#         if isinstance(tree, (ete3.Tree, ete3.PhyloTree, ete3.ClusterTree)):
#             # Check bifurcation
#             n_internal_nodes = len([*filter(lambda node:node.is_leaf() == False, tree.traverse())])
#             n_leaves = len([*filter(lambda node:node.is_leaf(), tree.traverse())])
#             if n_internal_nodes < (n_leaves - 1):
#                 tree.resolve_polytomy(**bifurcation_kws)
#                 if verbose:
#                     print("Resolving tree polytomy and forcing bifurcation", file=sys.stderr)
#             tree = ete_to_skbio(tree=tree, node_prefix=node_prefix)
#         return ilr_transform(table=X, tree=tree)
#     else:
#         return pd.DataFrame(ilr(X), index=X.index)
