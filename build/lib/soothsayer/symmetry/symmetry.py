
# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, multiprocessing, itertools, copy, datetime, warnings
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx

# Biology
from Bio import SeqIO, Seq
import ete3
import skbio

# SciPy
# from scipy import stats
from scipy.stats import entropy
from scipy.spatial.distance import squareform, pdist
from scipy.spatial import distance
try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage
    print("Could not import `linkage` from `fastcluster` and using `scipy.cluster.hierarchy.linkage` instead", file=sys.stderr)

# Statistics and Miscellaneous Machine Learning
# from astropy.stats import biweight_midcorrelation, median_absolute_deviation
from sklearn.metrics.pairwise import pairwise_distances

# Soothsayer
from ..transmute.conversion import linkage_to_newick
# from ..r_wrappers.packages.WGCNA import bicor
from ..utils import  infer_tree_type, is_symmetrical, force_symmetry, assert_acceptable_arguments, format_header, check_packages, is_dict_like, dict_build, is_number, is_nonstring_iterable, format_memory, name_tree_nodes, add_objects_to_globals

from ..io import write_object
import datetime, copy, warnings
from scipy.spatial.distance import squareform
from fastcluster import linkage
from soothsayer.transmute import linkage_to_newick
from skbio.util._decorator import experimental, stable
from scipy.stats import entropy

__all__ = {"pairwise", "pairwise_tree_distance","pairwise_difference", "pairwise_logfc"}

# compositional
import compositional as coda
functions_from_compositional = {"pairwise_vlr", "pairwise_rho","pairwise_phi"}
add_objects_to_globals(coda, functions_from_compositional, globals(), add_version=True, __all__=__all__)

# hive_networkx
import hive_networkx as hx
functions_from_hive_networkx= {}
add_objects_to_globals(hx, functions_from_hive_networkx, globals(), add_version=True, __all__=__all__)


# ensemble_networkx
import ensemble_networkx as enx
functions_from_ensemble_networkx= { "pairwise_biweight_midcorrelation", "Symmetric", "redundant_to_condensed", "condensed_to_redundant"}
add_objects_to_globals(enx, functions_from_ensemble_networkx, globals(), add_version=True, __all__=__all__)


__all__ = sorted(__all__)




# Pairwise interactions
def pairwise(X, metric="euclidean", axis=1, name=None, into=pd.DataFrame, association="infer", signed=True, n_jobs=-1, check_metrics=True, symmetric_kws=dict(), verbose=False):
    """
    Compute pairwise interactions

    01-August-2018
    """

    def get_data_object(df_dense, into, association, func_signed, metric, symmetric_kws, CORRELATION_METRICS, DISTANCE_METRICS, name, check_metrics):
        if hasattr(metric, "__call__"):
            metric_name = metric.__name__
        else:
            metric_name = metric

        if association == "similarity":
            if check_metrics:
                assert metric_name not in DISTANCE_METRICS, f"Cannot compute simlilarity for {metric.__name__}"

        if association == "dissimilarity":
            if metric_name in CORRELATION_METRICS:
                df_dense = 1 - func_signed(df_dense)
        if into in [Symmetric, pd.Series]:
            kernel = Symmetric(df_dense, association=association, name=name, **symmetric_kws)
            if into == Symmetric:
                return kernel
            if into == pd.Series:
                return kernel.data
        else:
            return df_dense


    fill_diagonal="auto"
    CORRELATION_METRICS = {"kendalltau", "pearson", "spearman", 'biweight_midcorrelation', 'bicor'}
    STATISTICAL_TEST_METRICS = {'ks_2samp', 'ttest_1samp', 'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel', "wilcoxon","ranksums", "mannwhitneyu"}
    DISTANCE_METRICS = distance.__dict__["__all__"]

    # Assertions
    accepted_associations = ["similarity", "dissimilarity", "statistical_test", "infer"]
    assert association in accepted_associations, f"{association} is not in {accepted_association}"

    # Metric name
    if hasattr(metric,"__call__"):
        metric_name = metric.__name__
    else:
        metric_name = metric
    # Infer association
    if association == "infer":
        if metric_name in CORRELATION_METRICS:
            association = "similarity"
        if metric_name in STATISTICAL_TEST_METRICS:
            association = "statistical_test"
        if metric_name in DISTANCE_METRICS:
            association = "dissimilarity"
        assert association != "infer", f"Unable to infer association from metric = {metric}"
        if verbose: 
            print(f"Inferred association as `{association}`", file=sys.stderr)

    # Similarity transformation function
    functions = {
        True: lambda X: (X+1)/2,
        False: lambda X:np.abs(X),
        None: lambda X:X,
    }
    func_signed = functions[signed]

    # Axis
    if axis == 0: X_copy = X.copy()
    if axis == 1: X_copy = X.copy().T

    # Locals
    Ar_X = X_copy.values
    labels = X_copy.index

    # Placeholder
    df_dense = None

    # Pairwise interactions
    while df_dense is None:
        # Statistical test
        if metric_name in STATISTICAL_TEST_METRICS:
            metric = lambda u,v: getattr(stats, metric_name)(u,v)[1]
            df_dense = pd.DataFrame(squareform(pdist(Ar_X, metric=metric)), index=labels, columns=labels)
            fill_diagonal = 0.0
            if association != "statistical_test":
                print(f"Convert `association` from `{association}` to `statistical_test`", file=sys.stderr)
                association = "statistical_test"
            break
        # Euclidean distance
        if metric_name == "euclidean":
            df_dense = pd.DataFrame(squareform(pdist(Ar_X, metric="euclidean")), index=labels, columns=labels)
            fill_diagonal = 0.0
            break
        # Biweight midcorrelation
        if metric in {"bicor", "biweight_midcorrelation"}:
            try:
                # df_dense = bicor(X_copy.T, condensed=False, self_interactions=True)
                df_dense = pairwise_biweight_midcorrelation(X_copy.T)
                break
            except:
                pass

        # Compute correlation and send the rest down the pipeline
        if metric_name in set(CORRELATION_METRICS) - set(["bicor", "biweight_midcorrelation"]):
            metric = metric_name
        if type(metric) == str:
            # Correlation
            if metric in CORRELATION_METRICS:
                df_dense = X_copy.T.corr(method = metric)
                break
            # Distance
            if metric_name in DISTANCE_METRICS:
                metric = getattr(distance, metric)

        # Check metric
        assert hasattr(metric, "__call__"), f"metric `{metric}` is unrecognized and should be converted into a callable at this point of the function"

        # Pairwise computation
        df_dense = pd.DataFrame(pairwise_distances(X_copy, metric=metric, n_jobs=n_jobs), index=labels, columns=labels)

        # Diagonal
        if fill_diagonal is not None:
            if fill_diagonal == "auto":
                fill_diagonal = {"similarity":1.0, "dissimilarity":0.0}[association]
            fill_diagonal = np.repeat(fill_diagonal, X_copy.shape[0]).tolist()
            np.fill_diagonal(df_dense.values, fill_diagonal)

    # Check dense
    assert df_dense is not None, f"`df_dense` is None.  Check the following: metric={metric}"

    # Check NaN
    df_isnull = df_dense.isnull()
    if np.any(df_isnull):
        num_null = df_isnull.values.ravel().sum()
        warnings.warn(f"There are {num_null} null values")

    # Symmetric keywords
    _symmetric_kws={
        "edge_type":metric_name,
        "func_metric":metric if hasattr(metric, "__call__") else None,
        "metadata":{
            "input_shape":X.shape,
        },
    }
    _symmetric_kws.update(symmetric_kws)

    # Return object
    return get_data_object(df_dense, into, association, func_signed, metric, _symmetric_kws, CORRELATION_METRICS, DISTANCE_METRICS, name, check_metrics)

# Pairwise distance on ete trees
def pairwise_tree_distance(tree, topology_only=True):
    d_topology = defaultdict(dict)
    leaves = tree.get_leaf_names()
    for i, id_obsv_A in enumerate(leaves):
        for j in range(i+1, len(leaves)):
            id_obsv_B = leaves[j]
            if id_obsv_A != id_obsv_B:
                d_topology[id_obsv_A][id_obsv_B] =  d_topology[id_obsv_B][id_obsv_A] = tree.get_distance(target=id_obsv_A, target2=id_obsv_B, topology_only=topology_only)
    df_dism_tree = pd.DataFrame(d_topology)
    np.fill_diagonal(df_dism_tree.values, 0)
    return df_dism_tree

# Pairwise difference
def pairwise_difference(X:pd.DataFrame, idx_ctrl, idx_treatment, name_ctrl="control", name_treatment="treatment"):
    """
    Positive values means they are greater in idx_treatment than in idx_ctrl.
    Negative values means they are greater in idx_ctrl than in idx_treatment.
    """
    # Init
    idx_attr = X.columns

    idx_ctrl = sorted(idx_ctrl)
    idx_treatment = sorted(idx_treatment)

    # Groups
    A_ctrl = X.loc[idx_ctrl,:].values
    A_treatment = X.loc[idx_treatment,:].values

    # Pairwise profiles
    diff_profiles = np.vstack(A_treatment[:, np.newaxis] - A_ctrl)

    # Labels
    idx_pairwise_labels = itertools.product(idx_treatment, idx_ctrl)
    names = [name_treatment, name_ctrl]
    return pd.DataFrame(diff_profiles, index=pd.MultiIndex.from_tuples(idx_pairwise_labels, names=names), columns=idx_attr)

# Pairwise log fold-change
def pairwise_logfc(X:pd.DataFrame, idx_ctrl, idx_treatment, func_log=np.log2, name_ctrl="control", name_treatment="treatment"):
    """
    Positive values means they are greater in idx_treatment than in idx_ctrl.
    Negative values means they are greater in idx_ctrl than in idx_treatment.
    """
    # return pairwise_difference(func_log(X), idx_ctrl=idx_ctrl, idx_treatment=idx_treatment, name_ctrl=name_ctrl, name_treatment=name_treatment) # MUST TEST THIS MORE FIRST BUT IT SHOULD WORK
    # Init
    idx_attr = X.columns

    idx_ctrl = sorted(idx_ctrl)
    idx_treatment = sorted(idx_treatment)

    # Log Transform
    X_logscale = func_log(X)

    # Groups
    A_ctrl = X_logscale.loc[idx_ctrl,:].values
    A_treatment = X_logscale.loc[idx_treatment,:].values

    # Pairwise profiles
    logfc_profiles = np.vstack(A_treatment[:, np.newaxis] - A_ctrl)

    # Labels
    idx_pairwise_labels = itertools.product(idx_treatment, idx_ctrl)
    names = [name_treatment, name_ctrl]
    return pd.DataFrame(logfc_profiles, index=pd.MultiIndex.from_tuples(idx_pairwise_labels, names=names), columns=idx_attr)
