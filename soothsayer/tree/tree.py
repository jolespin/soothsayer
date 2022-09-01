# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time
from collections import OrderedDict, defaultdict
# from scipy.cluster import hierarchy as sp_hierarchy
import networkx as nx
import pandas as pd
import ete3
import skbio
from ..utils import dataframe_to_matrixstring, name_tree_nodes
from ..transmute.conversion import ete_to_skbio, nx_to_ete

__all__ = ["create_tree"]
__all__ = sorted(__all__)

# ==============================================================================
# Trees
# ==============================================================================
# Create ete3 tree or skbio
def create_tree(
        # Base
        newick=None,
        name=None,
        format=0,
        dist=1.0,
        support=1.0,
        quoted_node_names=False,
        # ClusterTree
        text_array=None,
        fdist=None,
        # PhyloTree
        alignment=None,
        alg_format='fasta',
        sp_naming_function=None,
        # PhyloxmlTree
        phyloxml_clade=None,
        phyloxml_phylogeny=None,
        # Constructor
        node_prefix="y",
        into=ete3.Tree,
        prune=None,
        force_bifuraction=True,
        # Keywords
        tree_kws=dict(),
        bifurcation_kws=dict(recursive=True),
    ):
    """
    Next: Convert to NetworkX
    """
    # Should the tree be converted to skbio
    convert_to_skbio = False
    if into in [skbio.TreeNode]:
        into = ete3.Tree
        convert_to_skbio = True

    # ete3 construction
    if into == ete3.Tree:
        tree = ete3.Tree(newick=newick, format=format, quoted_node_names=quoted_node_names, **tree_kws)
    if into == ete3.ClusterTree:
        if isinstance(text_array, pd.DataFrame):
            text_array = dataframe_to_matrixstring(text_array)
        tree = ete3.ClusterTree(newick=newick, text_array=text_array, fdist=fdist, **tree_kws)
    if into == ete3.PhyloTree:
        tree = ete3.PhyloTree(newick=newick, alignment=alignment, alg_format=alg_format, sp_naming_function=sp_naming_function, format=format, **tree_kws)
    if into == ete3.PhyloxmlTree:
        tree = ete3.PhyloxmlTree(phyloxml_clade=phyloxml_clade, phyloxml_phylogeny=phyloxml_phylogeny,**tree_kws)

    # Set base attributes
    for k,v in dict(name=name, dist=dist, support=support).items():
        setattr(tree, k, v)

    # Prune
    if prune is not None:
        tree.prune(prune)

    # Bifurcation
    if force_bifuraction:
        n_internal_nodes = len([*filter(lambda node:node.is_leaf() == False, tree.traverse())])
        n_leaves = len([*filter(lambda node:node.is_leaf(), tree.traverse())])
        if n_internal_nodes < (n_leaves - 1):
            tree.resolve_polytomy(**bifurcation_kws)

    # Node prefix
    if node_prefix is not None:
        tree = name_tree_nodes(tree, node_prefix=node_prefix)
    if not convert_to_skbio:
        return tree
    # skbio
    else:
        return ete_to_skbio(tree, node_prefix=None)
