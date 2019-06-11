# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time
from collections import OrderedDict, defaultdict
from io import StringIO

# PyData
import pandas as pd
import numpy as np
import networkx as nx

# Tree
import ete3
import skbio

# SciPy
from scipy import stats
from scipy.cluster import hierarchy as sp_hierarchy
from scipy.spatial.distance import squareform
try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage
    print("Could not import `linkage` from `fastcluster` and using `scipy.cluster.hierarchy.linkage` instead", file=sys.stderr)

# Soothsayer
from ..utils import dict_reverse

__all__ = ["dism_to_linkage", "linkage_to_newick", "dism_to_tree", "name_ete_nodes", "ete_to_skbio", "nx_to_ete", "ete_to_nx"]

# Create clusters
def dism_to_linkage(df_dism, method="ward"):
    """
    Input: A (m x m) dissimilarity Pandas DataFrame object where the diagonal is 0
    Output: Hierarchical clustering encoded as a linkage matrix

    Further reading:
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.sp_hierarchy.linkage.html
    https://pypi.python.org/pypi/fastcluster
    """
    #Linkage Matrix
    dist = squareform(df_dism.values)
    return linkage(dist,method=method)

# Convert dendrogram to Newick
def linkage_to_newick(Z, labels):
    """
    Input :  Z = linkage matrix, labels = leaf labels
    Output:  Newick formatted tree string
    Edit: 2018-June-28
    https://github.com/biocore/scikit-bio/issues/1579
    """
    tree = sp_hierarchy.to_tree(Z, False) #scipy.sp_hierarchy.to_tree
    def build_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{(parentdist - node.dist)/2}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{(parentdist - node.dist)/2}{newick}"
            else:
                newick = ");"
            newick = build_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = build_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
            newick = f"({newick}"
            return newick
    return build_newick(tree, "", tree.dist, labels)

# Dissimilarity to ete
def dism_to_tree(df_dism:pd.DataFrame, into=ete3.Tree, method="ward", name=None):
    """
    https://en.wikipedia.org/wiki/Newick_format
    Input: pd.DataFrame distance/dissimilarity matrix, diagonal must be 0, must be symmetrical
    Output: ete3 Tree
    """
    # Check labels for tuples
    labels = df_dism.index
    encrypt=False
    if any(map(lambda x:type(x) == tuple, labels)):
        d_encode = dict(zip(labels, map(str,range(len(labels)))))
        d_decode = dict_reverse(d_encode)
        df_dism.index = df_dism.columns = labels = df_dism.index.map(lambda x: d_encode[x])
        encrypt = True
    # Linkage
    Z = linkage(df_dism, method=method)
    # Newick
    newick = linkage_to_newick(Z, labels=labels)
    tree = into(newick=newick)
    tree.name= name
    # Decode
    if encrypt:
        for node in tree.traverse():
            if node.is_leaf():
                node.name = d_decode[node.name]
    return tree

# Name ete3 nodes
def name_ete_nodes(tree,  node_prefix="y"):
    intermediate_node_index = 1
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = f"{node_prefix}{intermediate_node_index}"
            intermediate_node_index += 1
    return tree

# Convert ete3 to skbio
def ete_to_skbio(tree, node_prefix="y"):
    if node_prefix is not None:
        tree = name_ete_nodes(tree, node_prefix=node_prefix)
    return skbio.TreeNode.read(StringIO(tree.write(format=1, format_root_node=True)))

# Convert a NetworkX graph object to an ete3 Tree
def nx_to_ete(G, deepcopy=True, into=ete3.Tree, root="infer", verbose=False):
    # Add nodes
    subtrees = OrderedDict()
    for node_data in G.nodes(data=deepcopy):
        if deepcopy:
            node, node_attrs = node_data
            subtree = into()
            subtree.name = node # Had to do this becuase ClusterNode doesn't have `name` attribute in args
            subtree.add_features(**node_attrs)
        else:
            node = node_data
            subtree = into()
            subtree.name = node # Had to do this becuase ClusterNode doesn't have `name` attribute in args
        subtrees[node] = subtree

    # Add edges
    for edge_data in G.edges(data=deepcopy):
        if deepcopy:
            parent, child, attrs = edge_data
            subtrees[parent].add_child(child=subtrees[child])
            subtrees[parent].add_features(**{(child,k):v for k,v in node_attrs})
            subtrees[child].add_features(**{(parent,k):v for k,v in node_attrs})
        else:
            parent, child, attrs = edge_data
            subtrees[parent].add_child(child=subtrees[child])
    # Inferring the tree root
    if root == "infer":
        descedants = sorted(map(lambda subtree: (subtree, len(subtree.get_descendants())), subtrees.values()), key=lambda x:x[1])
        descendant_counts = list(map(lambda x:x[1], descedants))
        tree_root, num_descendants = descedants[-1]
        if descendant_counts.count(num_descendants) > 1:
            root = None
            if verbose:
                print(f"Warning: The root node could not be inferred because no node has a maximum number of descendants", file=sys.stderr )
        else:
            if verbose:
                print(f"{tree_root.name} was inferred as root with n={num_descendants} descendants", file=sys.stderr )
            return tree_root
    if root is not None:
        if root not in subtrees:
            if verbose:
                print(f"{tree_root.name} was not in the networkx graph {G.name}", file=sys.stderr )
            root = None
        else:
            return subtrees[root]
    # Returning dictionary of subtrees
    if root is None:
        return subtrees

# ete3 to NetworkX
def ete_to_nx(tree, create_using=None, node_prefix="infer"):
    # Get graph
    if create_using is None:
        graph = nx.OrderedDiGraph()
    else:
        graph = create_using
    # Add node prefixes
    if node_prefix == "infer":
        if "" in tree:
            node_prefix = "y"
        else:
            node_prefix = None
    if node_prefix is not None:
        tree = name_ete_nodes(tree, node_prefix=node_prefix)
    else:
        assert "" not in tree, "All nodes (including intermediate) must be named"
    # Construct graph
    for node in tree.traverse():
        for child in node.get_children()[::-1]:
            branch_length = tree.get_distance(node, child)
            graph.add_edge(node.name, child.name, weight=branch_length)
    return graph
