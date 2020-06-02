# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time, uuid

import pandas as pd
import numpy as np

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import check_packages

# from soothsayer.symmetry import *

# ==============================================================================
# R Imports
# ==============================================================================
if "rpy2" in sys.modules:
    from rpy2 import robjects as ro
    from rpy2 import rinterface as ri

    from rpy2.robjects.packages import importr
    try:
        from rpy2.rinterface import RRuntimeError
    except ImportError:
        from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects import pandas2ri
    R = ro.r
    NULL = ri.NULL
    #rinterface.set_writeconsole_regular(None)

# R Packages
# phyloseq = R_package_retrieve("phyloseq")
# philr = R_package_retrieve("philr")
# ape = R_package_retrieve("ape")

# Phylogenetic Isometric log ratio
@check_packages(["phyloseq", "philr", "ape"], language="r", import_into_backend=False)
def normalize_philr(X:pd.DataFrame, tree, node_prefix="y", part_weights = "uniform", ilr_weights = "uniform", verbose=True, bifurcation_kws=dict(recursive=True)):
    """
    tree should be some type of ete3 Tree
    """
    # Imports
    phyloseq = R_package_retrieve("phyloseq")
    philr = R_package_retrieve("philr")
    ape = R_package_retrieve("ape")

    assert isinstance(X, pd.DataFrame), "type(df_counts) must be pd.DataFrame"
    assert not np.any(X == 0), "`X` cannot contain any zero values because of the log-transform.  Give it a pseudocount. (X+1)"

    # Prune tree
    tree_set =  set(tree.get_leaf_names())
    leaf_set_from_X = set(X.columns)
    assert leaf_set_from_X <= tree_set, "`X` columns should be a subset of `tree` leaves"
    tree = tree.copy(method="deepcopy")
    if leaf_set_from_X < tree_set:
        n_before_pruning = len(tree_set)
        tree.prune(leaf_set_from_X)
        if verbose:
            print(f"Pruned {n_before_pruning - len(tree.get_leaves())} attributes to match X.columns", file=sys.stderr)

    # Check bifurcation
    n_internal_nodes = len([*filter(lambda node:node.is_leaf() == False, tree.traverse())])
    n_leaves = len([*filter(lambda node:node.is_leaf(), tree.traverse())])
    if n_internal_nodes < (n_leaves - 1):
        tree.resolve_polytomy(**bifurcation_kws)

    # Convert to R datatype
    r_df_counts = pandas_to_rpy2(X)
    # I/O into PhyloSeq object
    path_newick_temporary = uuid.uuid4().hex
    tree.write(format=1, outfile=path_newick_temporary)
    r_tree_phyloseq = phyloseq.read_tree(path_newick_temporary)
    os.remove(path_newick_temporary)
    # Relabel
    r_tree_phyloseq = ape.makeNodeLabel(r_tree_phyloseq, method="number", prefix=node_prefix)
    # Transform
    r_df_transform = philr.philr(df=R["as.matrix"](r_df_counts), tree=r_tree_phyloseq, part_weights=part_weights, ilr_weights=ilr_weights)
    return pd.DataFrame(rpy2_to_pandas(r_df_transform), index=X.index, columns=list(map(lambda x:f"{node_prefix}{x}", range(1,X.shape[1]))))
