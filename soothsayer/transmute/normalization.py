# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np

# Compositional
from skbio.stats.composition import clr, ilr
from gneiss.composition import ilr_transform
import ete3

# SciPy
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Soothsayer
from .conversion import ete_to_skbio
from ..r_wrappers.packages.edgeR import normalize_edgeR
from ..r_wrappers.packages.metagenomeSeq import normalize_css
# from soothsayer.r_wrappers.packages.philr import normalize_philr

from ..utils import is_dict, assert_acceptable_arguments

__all__ = ["normalize_minmax", "normalize_tss", "normalize_clr", "normalize_center", "normalize_zscore", "normalize_quantile", "normalize_boxcox", "normalize_ilr", "normalize", "normalize_expression"]

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

# CLR Normalization
def normalize_clr(X):
    assert isinstance(X, (pd.Series, pd.DataFrame)), "Type must be either pd.Series or pd.DataFrame"
    if isinstance(X, pd.DataFrame):
        A_clr = clr(X.values)
        return pd.DataFrame(A_clr, index=X.index, columns=X.columns)
    if isinstance(X, pd.Series):
        return pd.Series(clr(X),index=X.index, name=X.name)

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

# ILR Transformation
def normalize_ilr(X:pd.DataFrame, tree=None, node_prefix="y", verbose=True):
    """
    if `tree` is None then orthonormal basis for Aitchison simplex defaults to J.J.Egozcue orthonormal basis.
    """
    assert isinstance(X, pd.DataFrame), "type(X) must be pd.DataFrame"
    assert not np.any(X == 0), "`X` cannot contain any zero values because of the log-transform.  Give it a pseudocount. (X+1)"

    # Supply tree
    if tree is not None:
        tree_set =  set(tree.get_leaf_names())
        leaf_set_from_X = set(X.columns)
        assert leaf_set_from_X <= tree_set, "`X` columns should be a subset of `tree` leaves"
        tree = tree.copy(method="deepcopy")
        if leaf_set_from_X < tree_set:
            n_before_pruning = len(tree_set)
            tree.prune(leaf_set_from_X)
            if verbose:
                print(f"Pruned {n_before_pruning - len(tree.get_leaves())} attributes to match X.columns", file=sys.stderr)

        if isinstance(tree, (ete3.Tree, ete3.PhyloTree, ete3.ClusterTree)):
            # Check bifurcation
            n_internal_nodes = len([*filter(lambda node:node.is_leaf() == False, tree.traverse())])
            n_leaves = len([*filter(lambda node:node.is_leaf(), tree.traverse())])
            if n_internal_nodes < (n_leaves - 1):
                tree.resolve_polytomy(**bifurcation_kws)
                if verbose:
                    print("Resolving tree polytomy and forcing bifurcation", file=sys.stderr)
            tree = ete_to_skbio(tree=tree, node_prefix=node_prefix)
        return ilr_transform(table=X, tree=tree)
    else:
        return pd.DataFrame(ilr(X), index=X.index)



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
        df_normalized = normalize_clr(X)
    if method == "ilr":
        df_normalized = normalize_ilr(X, tree=tree)
    # if method == "philr":
    #     df_normalized = normalize_philr(X, tree=tree)

    # Transpose back
    if axis == 0:
        df_normalized = df_normalized.T
    return df_normalized

# Normalize gene expresssion data
def normalize_expression(X:pd.DataFrame, method:str="tpm", length:pd.Series=None, p=0.75, **kws):
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
