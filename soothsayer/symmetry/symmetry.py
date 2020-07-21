
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
functions_from_hive_networkx= {"Symmetric", "dense_to_condensed", "condensed_to_dense"}
add_objects_to_globals(hx, functions_from_hive_networkx, globals(), add_version=True, __all__=__all__)


# ensemble_networkx
import ensemble_networkx as enx
functions_from_ensemble_networkx= { "pairwise_biweight_midcorrelation"}
add_objects_to_globals(enx, functions_from_ensemble_networkx, globals(), add_version=True, __all__=__all__)


__all__ = sorted(__all__)



# =======================================================
# Pairwise calculations
# =======================================================
# def dense_to_condensed(X, name=None, assert_symmetry=True, tol=None):
#     if assert_symmetry:
#         assert is_symmetrical(X, tol=tol), "`X` is not symmetric with tol=`{}`".format(tol)
#     labels = X.index
#     index=pd.Index(list(map(frozenset,itertools.combinations(labels, 2))), name=name)
#     data = squareform(X, checks=False)
#     return pd.Series(data, index=index, name=name)

# def condensed_to_dense(y:pd.Series, fill_diagonal=np.nan, index=None):
#     # Need to optimize this
#     data = defaultdict(dict)
#     for edge, w in y.iteritems():
#         node_a, node_b = tuple(edge)
#         data[node_a][node_b] = data[node_b][node_a] = w
        
#     if is_dict_like(fill_diagonal):
#         for node in data:
#             data[node][node] = fill_diagonal[node]
#     else:
#         for node in data:
#             data[node][node] = fill_diagonal
            
#     df_dense = pd.DataFrame(data)
#     if index is not None:
#         df_dense = df_dense.loc[index,index]
#     return df_dense

# # Symmetrical dataframes represented as augment pd.Series
# @experimental(as_of="2020.06.23")
# class Symmetric(object):
#     """
#     An indexable symmetric matrix stored as the lower triangle for space

#     devel
#     =====
#     2020-June-23
#     * Replace self._dense_to_condensed to dense_to_condensed
#     * Dropped math operations
#     * Added input for Symmetric or pd.Series with a frozenset index

#     2018-August-16
#     * Added __add__, __sub__, etc.
#     * Removed conversion to dissimilarity for tree construction
#     * Added .iteritems method
    

#     Future:
#     * Use `weights` instead of `data`
    
#     Dropped:
#     Fix the diagonal arithmetic
#     """
#     def __init__(self, data, node_type=None, edge_type=None,  func_metric=None, name=None, association="infer", assert_symmetry=True, nans_ok=True, tol=None, acceptable_associations={"similarity", "dissimilarity", "statistical_test", "network", "infer", None}, **attrs):
        
#         self._acceptable_associations = acceptable_associations
#         # From Symmetric object
#         if isinstance(data, type(self)):
#             self.__dict__.update(data.__dict__)
#             if (self.func_metric is None) and (func_metric is not None):
#                 assert hasattr(func_metric, "__call__"), "`func_metric` isn't a function"
#                 if (self.edge_type is None) and (edge_type is None):
#                     edge_type = func_metric.__name__
#             if (self.node_type is None) and (node_type is not None):
#                 self.node_type = node_type
#             if (self.edge_type is None) and (edge_type is not None):
#                 self.edge_type = edge_type
#             if (association  not in {"infer", None}):
#                 assert_acceptable_arguments(association, self._acceptable_associations)
#                 self.association = association

#         else:
#             # From pd.DataFrame object
#             if isinstance(data, pd.DataFrame):
#                 self._from_dataframe(data=data, association=association, assert_symmetry=assert_symmetry, nans_ok=nans_ok, tol=tol)

#             # From pd.Series object
#             if isinstance(data, pd.Series):
#                 self._from_series(data=data, association=association)
                
#             # From pd.DataFrame or pd.Series
#             self.metadata = dict()
#             if not nans_ok:
#                 assert not np.any(data.isnull()), "Cannot move forward with missing values"
#             if func_metric is not None:
#                 assert hasattr(func_metric, "__call__"), "`func_metric` isn't a function"
#                 if edge_type is None:
#                     edge_type = func_metric.__name__
#             self.node_type = node_type
#             self.edge_type = edge_type
#             self.func_metric = func_metric
#             self.values = self.weights.values
#             self.number_of_nodes = self.nodes.size
#             self.number_of_edges = self.edges.size
            
#         # Universal
#         self.name = name
# #         self.graph = self.to_graph(into=graph)
#         self.memory = self.weights.memory_usage()
#         self.metadata.update(attrs)
#         self.__synthesized__ = datetime.datetime.utcnow()
                                      
 

#     # =======
#     # Utility
#     # =======
#     def _infer_association(self, X):
#         diagonal = np.diagonal(X)
#         diagonal_elements = set(diagonal)
#         assert len(diagonal_elements) == 1, "Cannot infer relationships from diagonal because multiple values"
#         assert diagonal_elements <= {0,1}, "Diagonal should be either 0.0 for dissimilarity or 1.0 for similarity"
#         return {0.0:"dissimilarity", 1.0:"similarity"}[list(diagonal_elements)[0]]

            
#     def _from_dataframe(self, data:pd.DataFrame, association, assert_symmetry, nans_ok, tol):
#         if assert_symmetry:
#             assert is_symmetrical(data, tol=tol), "`X` is not symmetric.  Consider dropping the `tol` to a value such as `1e-10` or using `(X+X.T)/2` to force symmetry"
#         assert_acceptable_arguments(association, self._acceptable_associations)
#         if association == "infer":
#             association = self._infer_association(data)
#         assert_acceptable_arguments(association, self._acceptable_associations)
#         self.association = association
#         self.nodes = pd.Index(sorted(data.index), name="Nodes")
#         self.diagonal = pd.Series(np.diagonal(data), index=data.index, name="Diagonal")[self.nodes]
#         self.weights = dense_to_condensed(data, name="Weights", assert_symmetry=assert_symmetry, tol=tol)
#         self.edges = pd.Index(self.weights.index, name="Edges")

                                      
#     def _from_series(self, data:pd.Series, association):
#         assert np.all(data.index.map(lambda edge: isinstance(edge, frozenset))), "If `data` is pd.Series then each key in the index must be a frozenset of size 2"
#         if association == "infer":
#             association = None
#         assert_acceptable_arguments(association, self._acceptable_associations)
#         self.association = association
#         self.weights = pd.Series(data, name="Weights")
#         self.edges = pd.Index(self.weights.index, name="Edges")
#         self.nodes = pd.Index(sorted(frozenset.union(*self.edges)), name="Nodes")

        
#     def set_diagonal(self, diagonal):
#         if diagonal is None:
#             self.diagonal = None
#         else:
#             if is_number(diagonal):
#                 diagonal = dict_build([(diagonal, self.nodes)])
#             assert is_dict_like(diagonal), "`diagonal` must be dict-like"
#             assert set(diagonal.keys()) >= set(self.nodes), "Not all `nodes` are in `diagonal`"
#             self.diagonal =  pd.Series(diagonal, name="Diagonal")[self.nodes]
            
#     # =======
#     # Built-in
#     # =======
#     def __repr__(self):
#         pad = 4
#         header = format_header("Symmetric(Name:{}, dtype: {})".format(self.name, self.weights.dtype),line_character="=")
#         n = len(header.split("\n")[0])
#         fields = [
#             header,
#             pad*" " + "* Number of nodes ({}): {}".format(self.node_type, self.number_of_nodes),
#             pad*" " + "* Number of edges ({}): {}".format(self.edge_type, self.number_of_edges),
#             pad*" " + "* Association: {}".format(self.association),
#             pad*" " + "* Memory: {}".format(format_memory(self.memory)),
#             *map(lambda line:pad*" " + line, format_header("| Weights", "-", n=n-pad).split("\n")),
#             *map(lambda line: pad*" " + line, repr(self.weights).split("\n")[1:-1]),
#             ]

#         return "\n".join(fields)
    
#     def __getitem__(self, key):
#         """
#         `key` can be a node or non-string iterable of edges
#         """

#         if is_nonstring_iterable(key):
#             assert len(key) >= 2, "`key` must have at least 2 identifiers. e.g. ('A','B')"
#             key = frozenset(key)
#             if len(key) == 1:
#                 return self.diagonal[list(key)[0]]
#             else:
#                 if len(key) > 2:
#                     key = list(map(frozenset, itertools.combinations(key, r=2)))
#                 return self.weights[key]
#         else:
#             if key in self.nodes:
#                 s = frozenset([key])
#                 mask = self.edges.map(lambda x: bool(s & x))
#                 return self.weights[mask]
#             else:
#                 raise KeyError("{} not in node list".format(key))
        
#     @experimental(as_of="2020.06.23")
#     def __call__(self, key, func=np.sum):
#         """
#         This can be used for connectivity in the context of networks but can be confusing with the versatiliy of __getitem__
#         """
#         if hasattr(key, "__call__"):
#             return self.weights.groupby(key).apply(func)
#         else:
#             return func(self[key])
        
#     def __len__(self):
#         return self.number_of_nodes
#     def __iter__(self):
#         for v in self.weights:
#             yield v
#     def items(self):
#         return self.weights.items()
#     def iteritems(self):
#         return self.weights.iteritems()
#     def keys(self):
#         return self.weights.keys()
    
#     def apply(self, func):
#         return func(self.weights)
#     def mean(self):
#         return self.weights.mean()
#     def median(self):
#         return self.weights.median()
#     def min(self):
#         return self.weights.min()
#     def max(self):
#         return self.weights.max()
#     def idxmin(self):
#         return self.weights.idxmin()
#     def idxmax(self):
#         return self.weights.idxmax()
#     def sum(self):
#         return self.weights.sum()
#     def sem(self):
#         return self.weights.sem()
#     def var(self):
#         return self.weights.var()
#     def std(self):
#         return self.weights.std()
#     def describe(self, **kwargs):
#         return self.weights.describe(**kwargs)
#     def map(self, func):
#         return self.weights.map(func)
#     def entropy(self, base=2):
#         assert np.all(self.weights > 0), "All weights must be greater than 0"
#         return entropy(self.weights, base=base)

# #     # Maths
# #     def __add__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data += x
# #         self.diagonal += x
# #         return symmetric_clone
# #     def __radd__(self, x):
# #         self.diagonal += x
# #         return Symmetric.__add__(self,x)
    
# #     def __sub__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data -= x
# #         return symmetric_clone
# #     def __rsub__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data = x - symmetric_clone.data
# #         return symmetric_clone
# #     def __mul__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data *= x
# #         return symmetric_clone
# #     def __rmul__(self, x):
# #         return Symmetric.__mul__(self,x)
# #     def __truediv__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data /= x
# #         return symmetric_clone
# #     def __rtruediv__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data = x / symmetric_clone.data
# #         return symmetric_clone
# #     def __floordiv__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data //= x
# #         return symmetric_clone
# #     def __rfloordiv__(self, x):
# #         symmetric_clone = copy.deepcopy(self)
# #         symmetric_clone.data = x // symmetric_clone.data
# #         return symmetric_clone

#     # ==========
#     # Conversion
#     # ==========
#     def to_dense(self, index=None):
#         return condensed_to_dense(y=self.weights, fill_diagonal=self.diagonal, index=index)

#     def to_condensed(self):
#         return self.weights

#     @check_packages(["ete3", "skbio"])
#     def to_tree(self, method="average", into=None, node_prefix="y"):
#         assert self.association == "dissimilarity", "`association` must be 'dissimilarity' to construct tree"
#         if method in {"centroid", "median", "ward"}:
#             warnings.warn("Methods ‘centroid’, ‘median’, and ‘ward’ are correctly defined only if Euclidean pairwise metric is used.\nSciPy Documentation - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage") 
#         if into is None:
#             into = ete3.Tree
#         if not hasattr(self,"Z"):
#             self.Z = linkage(self.weights.values, metric="precomputed", method=method)
#         if not hasattr(self,"newick"):
#             self.newick = linkage_to_newick(self.Z, self.nodes)
#         tree = into(newick=self.newick, name=self.name)
#         return name_tree_nodes(tree, node_prefix)

#     def to_networkx(self, into=None, **attrs):
#         if into is None:
#             into = nx.Graph
#         metadata = {"name":self.name, "node_type":self.node_type, "edge_type":self.edge_type, "func_metric":self.func_metric}
#         metadata.update(attrs)
#         graph = into(**metadata)
#         for (node_A, node_B), weight in self.weights.iteritems():
#             graph.add_edge(node_A, node_B, weight=weight)
#         return graph
    
#     def to_file(self, path, **kwargs):
#         write_object(obj=self, path=path, **kwargs)

#     def copy(self):
#         return copy.deepcopy(self)

# Pairwise interactions
def pairwise(X, metric="euclidean", axis=1, name=None, into=pd.DataFrame, association="infer", signed=True, n_jobs=-1, check_metrics=True, symmetric_kws=dict()):
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

# # Biweight midcorrelation
# def pairwise_biweight_midcorrelation(X, use_numba=False, verbose=False):
#     """
#     X: {np.array, pd.DataFrame}

#     Code adapted from the following sources:
#         * https://stackoverflow.com/questions/61090539/how-can-i-use-broadcasting-with-numpy-to-speed-up-this-correlation-calculation/61219867#61219867
#         * https://github.com/olgabot/pandas/blob/e8caf4c09e1a505eb3c88b475bc44d9389956585/pandas/core/nanops.py

#     Special thanks to the following people:
#         * @norok2 (https://stackoverflow.com/users/5218354/norok2) for optimization (vectorization and numba)
#         * @olgabot (https://github.com/olgabot) for NumPy implementation

#     Benchmarking:
#         * iris_features (4,4)
#             * numba: 159 ms ± 2.85 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#             * numpy: 276 µs ± 3.45 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#         * iris_samples: (150,150)
#             * numba: 150 ms ± 7.57 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#             * numpy: 686 µs ± 18.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

#     Future:
#         * Handle missing values

#     """
#     # Data
#     result = None
#     labels = None
#     if isinstance(X, pd.DataFrame):
#         labels = X.columns
#         X = X.values

#     def _base_computation(A):
#         n, m = A.shape
#         A = A - np.median(A, axis=0, keepdims=True)
#         v = 1 - (A / (9 * np.median(np.abs(A), axis=0, keepdims=True))) ** 2
#         est = A * v ** 2 * (v > 0)
#         norms = np.sqrt(np.sum(est ** 2, axis=0))
#         return n, m, est, norms

#     # Check if numba is available
#     assert_acceptable_arguments(use_numba, {True, False, "infer"})
#     if use_numba == "infer":
#         if "numba" in sys.modules:
#             use_numba = True
#         else:
#             use_numba = False
#         if verbose:
#             print("Numba is available:", use_numba, file=sys.stderr)

#     # Compute using numba
#     if use_numba:
#         assert "numba" in sys.modules
#         from numba import jit

#         def _biweight_midcorrelation_numba(A):
#             @jit
#             def _condensed_to_dense(n, m, est, norms, result):
#                 for i in range(m):
#                     for j in range(i + 1, m):
#                         x = 0
#                         for k in range(n):
#                             x += est[k, i] * est[k, j]
#                         result[i, j] = result[j, i] = x / norms[i] / norms[j]
#             n, m, est, norms = _base_computation(A)
#             result = np.empty((m, m))
#             np.fill_diagonal(result, 1.0)
#             _condensed_to_dense(n, m, est, norms, result)
#             return result

#         result = _biweight_midcorrelation_numba(X)
#     # Compute using numpy
#     else:
#         def _biweight_midcorrelation_numpy(A):
#             n, m, est, norms = _base_computation(A)
#             return np.einsum('mi,mj->ij', est, est) / norms[:, None] / norms[None, :]
#         result = _biweight_midcorrelation_numpy(X)

#     # Add labels
#     if labels is not None:
#         result = pd.DataFrame(result, index=labels, columns=labels)

#     return result
