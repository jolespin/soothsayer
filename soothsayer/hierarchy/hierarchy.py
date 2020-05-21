
# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, copy, warnings, pathlib
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np
import networkx as nx

# Biology
from Bio import SeqIO, Seq
import ete3
import skbio

# SciPy
from scipy.cluster import hierarchy as sp_hierarchy
try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage
    warnings.warn("Could not import `linkage` from `fastcluster` and using `scipy.cluster.hierarchy.linkage` instead")

## Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Soothsayer
from ..io import write_object
from ..r_wrappers.packages.dynamicTreeCut import cutree_dynamic
from ..utils import *
from ..ordination import PrincipalComponentAnalysis, eigenprofiles_from_data
from ..symmetry import Symmetric, pairwise
from ..transmute.conversion import linkage_to_newick, dism_to_linkage, ete_to_nx
from ..networks import intramodular_connectivity
from ..tree import create_tree


# Hierarchical Clustering
def _func_dcoord(dcoord):
    return -np.log(dcoord+1)
class Agglomerative(object):
    # Sycamore, Ent, Cypress, Telperion, Bifurcation, Echelon, Yucca, Nimloth
    """
    # ==========
    # __future__
    # ==========
    * Incorporate missing values for track plots [BUG]
    * Add markers/annotations for specific nodes
    * Polar functionality for the following:
        * horizontal_lines
        * _plot_clusters_mpl
        * _plot_tracks_mpl
    # ==========
    # devel
    # ==========
    2019-May-29
    * Added spines to tracks
    * Added bar outline
    2019-March-11
    * Removed `cluster_mapping`
    * Added `add_secondary_classes` like `add_tracks`
    * Adapted `_plot_clusters_mpl` to plotting secondary classes as well
    * Changed `_plot_clusters_mpl` -> `_plot_classes_mpl`

    2019-February-22
    * Add label encoder
    * Clean up axes
    * Change `extract` to `get`
    * Add polar projection dendrogram
    * cdict and ndict -> *_colors and node_ampping
    * cluster_assignment -> cluster_mapping
    * Removed dbscan compatibility
    ___________________________________________________________________________________
    ___________________________________________________________________________________
    2018-August-16
    * Added ability to use Symmetric objects
    * Added as_graph method
    ___________________________________________________________________________________
    ___________________________________________________________________________________
    2018-August-04
    * Fixed case where node names have incompatible characters with newick strings
    ___________________________________________________________________________________
    # =====
    # Usage
    # =====
    X_modified = (X_iris + np.random.RandomState(0).normal(loc=10, size=(X_iris.shape)))
    df_dism = pairwise(X_modified, "euclidean", 0)

    # Instantiate the clustering object
    x = Agglomerative(df_dism, data=X_modified, name="iris", leaf_type="subject", tree_type=ete3.ClusterTree)
    # Hierarchical clustering using R's cuttreedynamic function (like WGCNA)
    x.cluster()
    # Add extra data to plot on dendrogram
    x.add_track("petal-length_1", X_iris["petal_length"], color=c_iris, plot_type="bar")
    x.add_track("petal-length_2", X_iris["petal_length"], color="black", plot_type="area")
    x.add_track("sepal-length", X_iris["sepal_length"], color=c_iris, plot_type="bar")
    # Plot the data
    x.plot(show_tracks=True, show_leaves=False, cluster_size=20, distance_label="euclidean distance", title="iris dataset w/ N(10,1) noise")
    print(x.silhouette(1))
    # ==========
    # Notes
    # ==========
    # Since ete3 trees are used and the node names can be limited using newick format, the leaf names are relabeled. In particular, check out _relabel and the linkage_to_newick function... it will make sense.
    # ==========
    # Known bugs
    # ==========
    *  Missing values causes track plotting to break
         * x.add_track("sepal-length-species_A", X_iris["sepal_length"][:50], color=c_iris[:50], plot_type="line", ylim=(X_iris["sepal_length"].min(),X_iris["sepal_length"].max()))
    * When using `ylim` with `add_track` the plot comes back as empty :/
    * Leaves must be strings for some reason...
    """
    def __init__(self,
                 kernel,
                 name=None,
                 data=None,
                 leaf_axis="infer",
                 method="ward",
                 leaf_type=None,
                 tree_type=ete3.ClusterTree,
                 palette="hls",
                 outlier_color="white",
                 metadata=dict(),
                 tree_kws=dict(),
                 polar_kws=dict(),
                 rectilinear_kws=dict(),
                ):
        # Check the kernel object type
        accepted_kernel_types = {pd.DataFrame, Symmetric}
        assert type(kernel) in accepted_kernel_types, f"`kernel` type must be one of the following: {accepted_kernel_types}"

        # pd.DataFrame -> Symmetric
        if isinstance(kernel, pd.DataFrame):
            kernel = Symmetric(X=kernel, data_type=leaf_type, name=name, mode="dissimilarity", metadata=metadata)

        # Base
        self.name = name
        self.data = data
        self.kernel = kernel

        # Clustering
        self.method = method
        self.linkage_labels = self.kernel.labels.copy()
        self.num_leaves = len(self.linkage_labels)
        self._relabel = dict(zip(range(self.num_leaves), self.linkage_labels))
        self.Z = linkage(self.kernel.data.values, method=method)
        if leaf_axis == "infer":
            if data is not None:
                axis_0_overlap = len(set(self.linkage_labels) & set(data.index))
                axis_1_overlap = len(set(self.linkage_labels) & set(data.columns))
                leaf_axis = np.argmax([axis_0_overlap, axis_1_overlap])
            else:
                leaf_axis = None
        self.leaf_axis = leaf_axis
        self._clusters_exist = False

        # Data
        self.tree_type = tree_type
        if tree_type == ete3.ClusterTree:
            if data is not None:
                if leaf_axis == 0:
                    matrix_string = dataframe_to_matrixstring(data)
                elif leaf_axis == 1:
                    matrix_string = dataframe_to_matrixstring(data.T)
            else:
                matrix_string = None
            _tree_kws = {"text_array":matrix_string}
        else:
            _tree_kws = {}
        _tree_kws.update(tree_kws)

        #Tree
        self.newick = linkage_to_newick(self.Z, range(self.num_leaves))
        self.tree = tree_type(newick=self.newick, **_tree_kws)
        self.tree.name = name # Hack for using ete3.ClusterTree objects b/c there's no name argument
        self._relabel_tree(self.tree, self._relabel)
        name_tree_nodes(self.tree, node_prefix="y")
        self.dendrogram = sp_hierarchy.dendrogram(self.Z, labels=self.linkage_labels, no_plot=True)

        # Colors
        self.palette = palette
        self.outlier_color = outlier_color

        # Polar

        self._polar_kws = {"pad":0.02, "n_smooth":100, "func_dcoord":_func_dcoord}
        self._polar_kws.update(polar_kws)

        # Rectilinear
        self._rectilinear_kws = {}
        self._rectilinear_kws.update(rectilinear_kws)

        # Leaves
        self.leaf_type = leaf_type
        self.leaves = pd.Index(self.dendrogram["ivl"], name=leaf_type)
        self.leaf_ticks_rectilinear = np.arange(5, self.num_leaves * 10 + 5, 10)
        t_grid = np.linspace(self._polar_kws["pad"]/2, 1-self._polar_kws["pad"]/2, self.num_leaves)
        self.leaf_ticks_polar = 2*np.pi*t_grid


        self.leaf_positions_rectilinear = dict(zip(self.leaves,self.leaf_ticks_rectilinear))
        self.leaf_positions_polar = dict(zip(self.leaves, self.leaf_ticks_polar))

        # Extension data
        self.tracks = OrderedDict()
        self.secondary_classes = OrderedDict()
        self.axes = list()

        # Metadata
        self.__synthesized__ = datetime.datetime.utcnow()
        self.metadata = {
            "name":self.name,
            "method":self.method,
            "num_leaves":self.num_leaves,
            "leaf_type":leaf_type,
            "tree_type":tree_type.__name__,
            "synthesized":self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S"),
            "num_tracks":0}
        self.metadata.update(metadata)

        # Leaf attributes
        self.leaf_attributes = pd.DataFrame([
            pd.Series(self.leaf_positions_rectilinear, name="tick_position(rectilinear)"),
            pd.Series(self.leaf_positions_polar, name="tick_position(polar)"),
        ]).T.loc[self.leaves,:]
        self.leaf_attributes.index.name = self.leaf_type
    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.metadata)[1:-1]})"

    def copy(self):
        return copy.deepcopy(self)



    # =======
    # Conversion
    # =======
    def _relabel_tree(self, tree, _relabel):
        for node in tree.traverse():
            if node.is_leaf():
                node.name = _relabel[int(node.name)]
        return tree

    # =======
    # Utility
    # =======
    def to_file(self, path, compression="infer"):
        write_object(self, path, compression=compression, serialization_module=pickle, protocol=pickle.HIGHEST_PROTOCOL)

    # ==========
    # Clustering
    # ==========
    def cluster(self, deep_split=2, min_c=3, cut_method="hybrid", cut_height=None, algo_kws=dict(), outlier_module=None):
        """
        Future: Cluster using the actual ete3 nodes
        """
        accepted_arguments = {"hybrid", "static"}
        assert cut_method in accepted_arguments, f"`cut_method` must be one of the following: {accepted_arguments}"

        # Utility
        def _process_subtrees(tree, clusters):
            d_cluster_node = dict(zip(clusters.keys(), [np.nan]*len(clusters)))
            for cluster, leaves in clusters.iteritems():
                subtree = tree.get_common_ancestor(*leaves)
                d_cluster_node[cluster] = subtree.name
            return d_cluster_node

        def _process_static_clusters(cluster_mapping, min_c, outlier_module):
            # Sort clusters by size
            sorted_clusters = sorted(pd_series_collapse(cluster_mapping, type_collection=pd.Index).values, key= len, reverse=True)
            # Group the outliers
            d_cluster_leaves = defaultdict(list)

            i = 0
            for cluster in sorted_clusters:
                # WIP
                if len(cluster) < min_c: # There will never be clusters smaller than this from cutreedynamic
                    d_cluster_leaves[outlier_module] += cluster.tolist()
                else:
                    d_cluster_leaves[i] = cluster.tolist()
                    i += 1
            return d_cluster_leaves

        cluster_names = np.nan
        outlier_module = -1
        # Static tree cutting
        if cut_method == "static":
            _algo_kws = {"criterion":"distance"}
            _algo_kws.update(algo_kws)
            # Cut tree at threshold and get clusters indices
            clusters = sp_hierarchy.fcluster(Z=self.Z, t=cut_height, **_algo_kws)
            cluster_mapping = pd.Series(clusters, index=self.linkage_labels)
            # Relabel and identify outliers
            d_cluster_leaves = _process_static_clusters(cluster_mapping, min_c=min_c, outlier_module=outlier_module)
            cluster_mapping = pd.Series(dict_expand(d_cluster_leaves, into=OrderedDict), name=self.name).sort_values()

        # Hybrid Dynamic Tree Cutting
        if cut_method == "hybrid":
            _algo_kws = {}
            _algo_kws.update(algo_kws)
            cluster_mapping = cutree_dynamic(self.kernel, cut_method=cut_method, method=self.method, minClusterSize=min_c, deepSplit=deep_split, name=self.name, **_algo_kws).astype(int) #Outlier module

        # Index the clusters by the leaves
        self.cluster_mapping = cluster_mapping[self.leaves]

        # Set the colors
        self.colors = Chromatic.from_classes(cluster_mapping, name=self.name, palette=self.palette, format="hex", outlier_color=self.outlier_color, outlier_value=outlier_module)
        self.leaf_colors = self.colors.obsv_colors
        self.cluster_colors = self.colors.class_colors


        # Global objects
        # ==============
        # Add colors and clusters
        self.leaf_attributes["cluster"] = self.cluster_mapping
        self.leaf_attributes["color"] = self.leaf_colors
        # Add nodes
        self.node_mapping = _process_subtrees(self.tree, self.get_clusters(grouped=True))

        # Update
        self.cut_method = cut_method
        if cut_method == "static":
            self.cut_height = cut_height
        self.outlier_module = outlier_module
        self.num_clusters = len(self.cluster_mapping[self.cluster_mapping.map(lambda x:x != outlier_module)].unique())
        self.clusters = sorted(self.get_clusters(grouped=True).keys())
        self.metadata["num_clusters"] = self.num_clusters
        self._clusters_exist = True
        self._outlier_module_exists = outlier_module in self.cluster_mapping.index

        # Cluster attributes
        self.cluster_attributes = pd.DataFrame([
            pd.Series(self.cluster_colors, name="color"),
            pd.Series(self.node_mapping, name="node"),
            pd.Series(self.get_clusters(grouped=True).map(set), name="leaves"),
        ]).T
        self.cluster_attributes["size"] = self.cluster_attributes["leaves"].map(len)
        self.cluster_attributes.index.name = "id_cluster"
        return self
    # ======
    # Groups
    # ======
    def add_secondary_class(self, name, mapping:pd.Series, class_colors=None, pad=0.1, percent_of_figure = 20, palette="hls", desat=None):
        mapping = pd.Series(mapping)
        missing_leaves = set(self.leaves) - set(mapping.index)
        assert len(missing_leaves) == 0, "`mapping` does have an secondary class for each leaf.  In particular, the following:\n{}".format(missing_leaves)
        self.leaf_attributes[name] = mapping
        if class_colors is None:
            class_colors = map_colors(mapping.unique(),mode=0, palette=palette, desat=desat)
        self.secondary_classes[name] = {"mapping":mapping, "class_colors":class_colors, "pad":pad, "size":f"{percent_of_figure}%"}
        return self

    # ======
    # Tracks
    # ======
    def add_track(self, name, data, alpha=0.618, color="black", pad=0.25, percent_of_figure = 20, ylim=None, plot_type="bar", spines=["top", "bottom", "left", "right"], fillna=0, plot_kws=dict(), label_kws=dict(),  bar_width="auto", check_missing_values=False, convert_to_hex=False, horizontal_kws=dict(), horizontal_lines=[]):
        """
        type(track) should be a pd.Series
        """
        # Keywords
        if plot_type == "bar":
            _plot_kws = {"linewidth":1.382, "alpha":alpha, "width":max(5,self.num_leaves/30) if bar_width == "auto" else bar_width}
        elif plot_type == "area":
            _plot_kws = {"linewidth":0.618, "alpha":0.5}
        else:
            _plot_kws = {}
        _plot_kws.update(plot_kws)
        _horizontal_kws = {"linewidth":1, "linestyle":"-", "color":"black"}
        _horizontal_kws.update(horizontal_kws)

        # Color of the track
        if color is None:
            color = "black"
        if isinstance(color,str):
            color = pd.Series([color]*len(self.leaves), index=self.leaves)
        if is_dict(color):
            color = pd.Series(color, index=index)
        if convert_to_hex:
            _sampleitem_ = color[0]
            assert mpl.colors.is_color_like(_sampleitem_), f"{_sampleitem_} is not color-like"
            if is_rgb_like(_sampleitem_):
                color = color.map(rgb2hex)
        # Check for missing values
        if check_missing_values:
            assert set(data.index) == set(self.leaves), "`data.index` is not the same set as `self.leaves`"
            assert set(color.index) == set(self.leaves), "`color.index` is not the same set as `self.leaves`"
        if plot_type in ["area","line"]:
            if color.nunique() > 1:
                print("Warning: `area` and `line` plots can only be one color.  Defaults to first color in iterable.", file=sys.stderr)
        # Horizontal lines
        if is_number(horizontal_lines):
            horizontal_lines = [horizontal_lines]
        # Convert to ticks
        data = data[self.leaves]
        color = color[self.leaves]
        # Store data
        self.tracks[name] = {"data":data, "color":color, "pad":pad, "size":f"{percent_of_figure}%", "plot_type":plot_type, "plot_kws":_plot_kws,  "ylim":ylim, "horizontal_lines":horizontal_lines, "horizontal_kws":_horizontal_kws, "spines":spines}
        self.metadata["num_tracks"] += 1

        return self

    # ============
    # get Data
    # ============
    def _get(self, query, field, grouped):
        try:
            if not grouped:
                if query is None:
                    return self.leaf_attributes[field]

                else:
                    if is_nonstring_iterable(query):
                        mask = self.leaf_attributes[field].map(lambda x: x in query)
                        idx_leaves_query = self.leaf_attributes[field][mask].index
                        idx_leaves_query.name = tuple(sorted(query))
                        return idx_leaves_query

                    else:
                        return self.leaf_attributes[field][lambda x: x == query].index

            else:
                return pd_series_collapse(self.leaf_attributes[field], type_collection=pd.Index)

        except AttributeError:
            print("AttributeError: You must define or calculate the clusters first with `cluster` method", file=sys.stderr)
            return None

    # Clusters
    def get_clusters(self, query=None, grouped=False):
        return self._get(query=query, field="cluster", grouped=grouped)

    # Colors
    def get_colors(self, query=None, grouped=False):
        return self._get(query=query, field="color", grouped=grouped)

    # Sizes
    def get_sizes(self, query=None):
        if not is_nonstring_iterable(query):
            query = [query]
        return self.cluster_attributes.loc[slice(*query),"size"]

    # Nodes
    def get_nodes(self, query=None):
        if not is_nonstring_iterable(query):
            query = [query]
        return self.cluster_attributes.loc[slice(*query),"node"]

    # Subtrees
    def get_subtrees(self, query=None):
        def _search_tree(tree, query):
            node_name = self.node_mapping[query]
            return tree.search_nodes(name=node_name)[0]
        if query is not None:
            return _search_tree(self.tree, query)
        else:
            return pd.Series([*map(lambda x:_search_tree(self.tree, x), self.clusters)], index=pd.Index(self.clusters, name="subtrees"))
    # Tracks
    def get_tracks(self, query=None):
        if len(self.tracks) > 0:
            if query is not None:
                return self.tracks[query]
            else:
                df_tracks = pd.DataFrame(OrderedDict([(name, values["data"]) for name, values in self.tracks.items()]))
                df_tracks.index = self.leaves
                return df_tracks
    # ============
    # Plotting
    # ============
    def _plot_dendrogram_mpl(self, icoord, dcoord, xlim, ylim, color, ax, line_kws, n_smooth, distance_tick_kws, distance_label, show_distance, horizontal_lines, vertical_lines, func_dcoord):
        # Smoothing segments for polar plotting
        def _smooth_segment(seg, n_smooth):
            # https://stackoverflow.com/questions/51936574/how-to-plot-scipy-hierarchy-dendrogram-using-polar-coordinates
            # Note, this and the polar plotting has been adapted from above
            # Acknowledgements to @chthonicdaemon ^^^
            return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], n_smooth), [seg[3]]])

        # Plot dendrogram
        for xs, ys in zip(icoord, dcoord):
            if ax.name == "polar":
                xs = _smooth_segment(xs, n_smooth=n_smooth)
                ys = _smooth_segment(ys, n_smooth=n_smooth)
            ax.plot(xs, ys,  color, **line_kws)

        # Add vertical lines
        if bool(vertical_lines):
            if is_nonstring_iterable(vertical_lines) and not is_dict(vertical_lines):
                vertical_lines = {pos:dict() for pos in vertical_lines}
            for pos, kws in vertical_lines.items():
                ax.axvline(pos, **kws)
        # Add horizontal lines
        if bool(horizontal_lines):
            if ax.name == "polar":
                print("Warning: `horizontal_lines` visualization is currently not available with polar projection", file=sys.stderr)
            else:
                if is_nonstring_iterable(horizontal_lines) and not is_dict(horizontal_lines):
                    horizontal_lines = {pos:dict() for pos in horizontal_lines}
                for pos, kws in horizontal_lines.items():
                    ax.axhline(func_dcoord(pos), **kws)


        # Adjust ticks
        if ax.name == "polar":
            ax.set_rlabel_position(0)
            ax.set_xticks(self.leaf_ticks_polar)
        else:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            #Distance label
            conditions = dict()
            conditions[1] = distance_label is not None
            conditions[2] = show_distance
            if all(conditions.values()):
                ax.set_ylabel(distance_label, **distance_tick_kws)
        if not show_distance:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

        return ax



    def _plot_track_mpl(self, name, divider, xlim, ylim, show_track_ticks, show_track_labels, label_kws):
        ax = divider.append_axes("bottom", size=self.tracks[name]["size"], pad=self.tracks[name]["pad"])
        data = self.tracks[name]["data"]
        c = self.tracks[name]["color"]
        kws = self.tracks[name]["plot_kws"]
        # Set leaf tick positions
        leaf_positions = self.leaf_positions_rectilinear #leaf_positions = {True:self.leaf_positions_polar, False:self.leaf_positions_rectilinear}[polar]
        data.index = data.index.map(lambda x:leaf_positions[x])
        c.index = c.index.map(lambda x:leaf_positions[x])

        if self.tracks[name]["plot_type"] == "bar":
            ax.bar(data.index, data.values, color=c, **kws)
            kws_outline = kws.copy()
            del kws_outline["alpha"]
            ax.bar(data.index, data.values, color="none", edgecolor=c, **kws_outline)
        else:
            if self.tracks[name]["plot_type"] == "area":
                data.plot(kind="area", color=c.values, ax=ax, **kws)
                data.plot(kind="line", color=c.values, alpha=1, ax=ax)
            else:
                data.plot(kind=self.tracks[name]["plot_type"], color=c.values, ax=ax,  **kws)
        if len(self.tracks[name]["horizontal_lines"]):
            for y_pos in self.tracks[name]["horizontal_lines"]:
                ax.axhline(y_pos, **self.tracks[name]["horizontal_kws"])
        # Spines
        for spine in ["top", "bottom", "left", "right"]:
            if spine not in self.tracks[name]["spines"]:
                ax.spines[spine].set_visible(False)
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        if self.tracks[name]["ylim"] is not None:
            ax.set_ylim(ylim)
        if not show_track_ticks:
            ax.set_yticklabels([])
        if show_track_labels:
            ax.set_ylabel(name, **label_kws, ha="right", va="center")
        return ax


    def _plot_classes_mpl(self, size, pad, divider, xlim, show_cluster_labels, fill_kws, cluster_kws, name, track_label, label_kws, show_track_labels):
        ax = divider.append_axes("bottom", size=size, pad=pad)
        mode = {True:"primary", False:"secondary"}[name is None]
        if mode == "primary":
            mapping = self.leaf_attributes["cluster"]
            class_colors = self.cluster_colors

            # Labels
            if show_cluster_labels:
                for cluster, leaves in self.get_clusters(grouped=True).iteritems():
                    idx_middleleaf = leaves[len(leaves)//2]
                    label_position = self.leaf_positions_rectilinear[idx_middleleaf]
                    label = cluster
                    ax.annotate(label, xy=(label_position, 0.5), **cluster_kws)
        elif mode == "secondary":
            mapping = self.secondary_classes[name]["mapping"]
            class_colors = self.secondary_classes[name]["class_colors"]

        for leaf, position in self.leaf_positions_rectilinear.items():
            color = class_colors[mapping[leaf]]
            # Cluster rectangle
            ax.fill_between(x=[position - 5,position + 5], y1=0, y2=1, color=color, **fill_kws)

        if all([show_track_labels, (track_label is not None)]):
            ax.set_ylabel(track_label, **label_kws, ha="right", va="center")

        ax.set_xlim(xlim)
        ax.set_yticklabels([])
        return ax


    # Plot controller
    def plot(self,
             branch_color="auto",
             leaf_colors=None,
             leaf_font_size=15,
             leaf_rotation="auto",
             style="seaborn-white",
             show_leaves=False,
             show_clusters=True,
             show_cluster_labels=True,
             show_secondary_classes=True,
             show_tracks=True,
             show_track_ticks=False,
             show_track_labels=True,
             show_spines=True,
             show_xgrid=False,
             show_ygrid=False,
             show_distance=True,
             cluster_size=20,
             cluster_pad=0.1,
             cluster_label="cluster",
             figsize=(21,5),
             title=None,
             ax=None,
             legend=False,
             fig_kws=dict(),
             legend_kws=dict(),
             label_kws=dict(),
             line_kws=dict(),
             fill_kws=dict(),
             cluster_kws=dict(),
             leaf_tick_kws=dict(),
             distance_tick_kws=dict(),
             distance_label="Distance",
             title_kws=dict(),
             polar=False,
             pad=None,
             horizontal_lines=dict(),
             vertical_lines=dict(),
            ):

        # Initialize
        self.axes = list()
        orientation = "top" # Does not work with bottom, left, or right yet 2018-June-26

        # Keywords
        _fig_kws = {"figsize":figsize}
        _fig_kws.update(fig_kws)
        _label_kws = {"fontsize":15, "rotation":0, "horizontalalignment":"right"}
        _label_kws.update(label_kws)
        _legend_kws = {'fontsize': 12, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
        _legend_kws.update(legend_kws)
        _line_kws = {"linestyle":"-", "linewidth":1.382}
        _line_kws.update(line_kws)
        _fill_kws = {"alpha":0.618}
        _fill_kws.update(fill_kws)
        _cluster_kws = {"fontsize":12, "verticalalignment":"center", "horizontalalignment":"center", "bbox":dict(boxstyle="square", fc="w", pad=0.25, alpha=0.85)}
        _cluster_kws.update(cluster_kws)
        _distance_tick_kws = {"fontsize":15, "rotation":{True:0, False:90}[all([show_tracks, self.metadata["num_tracks"] > 0])]}
        if _distance_tick_kws["rotation"] == 0:
            _distance_tick_kws["horizontalalignment"] = "right"
        _distance_tick_kws.update(distance_tick_kws)
        if leaf_rotation in ["auto",None]:
            leaf_rotation = {True:90, False:0}[orientation in ["top","bottom"]]
        else:
            leaf_rotation = leaf_rotation
        _leaf_tick_kws = {"fontsize":12, "rotation":leaf_rotation}
        _leaf_tick_kws.update(leaf_tick_kws)
        _title_kws = {"fontsize":18, "fontweight":"bold","y":0.95}
        _title_kws.update(title_kws)


        # Branch colors
        if branch_color == "auto":
            branch_color = {True:"white", False:"black"}[style == "dark_background"]

        # Legend
        if legend == False:
            legend = None

        if legend is not None:
            if legend == True:
                if hasattr(self, "cluster_colors"):
                    legend = self.cluster_colors
            else:
                if isinstance(legend, pd.Series):
                    legend = legend.to_dict(OrderedDict)
        # Adjust coordinates for polar
        icoord=np.asarray(self.dendrogram["icoord"], dtype=float)
        dcoord=np.asarray(self.dendrogram["dcoord"], dtype=float)

        func_dcoord = lambda dcoord: dcoord

        if polar:
            func_dcoord = self._polar_kws["func_dcoord"]
            dcoord = func_dcoord(dcoord)
            icoord_max = icoord.max()
            icoord_min = icoord.min()
            icoord = ((icoord - icoord_min)/(icoord_max - icoord_min) * (1 - self._polar_kws["pad"]) + self._polar_kws["pad"]/2)*2*np.pi
        else:
            if "func_dcoord" in self._rectilinear_kws:
                func_dcoord = self._rectilinear_kws["func_dcoord"]
                dcoord = func_dcoord(dcoord)

        # Limits
        # Independent variable plot width
        tree_width = len(self.leaves) * 10
        xlim = (0,tree_width)
            # Depenendent variable plot height
        maximum_height = dcoord.flatten().max() #max(self.Z[:, 2])
        tree_height = maximum_height + maximum_height * 0.05
        ylim = (0,tree_height)

        # Plotting
        with plt.style.context(style):
            if ax is None:
                fig = plt.figure(**_fig_kws)
                ax_dendrogram = fig.add_subplot(111, polar=polar)
            else:
                fig = plt.gcf()
                ax_dendrogram = ax

            # Arguments for dendrogram plotting
            args_dendro = dict(
                icoord=icoord,
                dcoord=dcoord,
                color=branch_color,
                xlim=xlim,
                ylim=ylim,
                ax=ax_dendrogram,
                line_kws=_line_kws,
                distance_tick_kws=_distance_tick_kws,
                distance_label=distance_label,
                n_smooth=self._polar_kws["n_smooth"],
                show_distance=show_distance,
                horizontal_lines=horizontal_lines,
                vertical_lines=vertical_lines,
                func_dcoord=func_dcoord,
            )
            ax_dendrogram = self._plot_dendrogram_mpl(**args_dendro)
            self.axes.append(ax_dendrogram)
            # Add tracks
            divider = make_axes_locatable(ax_dendrogram)
            if show_tracks:
                if not polar:
                    if len(self.tracks) > 0:

                        for name in self.tracks:
                            if "edgecolor" in self.tracks[name]["plot_kws"]:
                                if self.tracks[name]["plot_kws"]["edgecolor"] == "auto":
                                    self.tracks[name]["plot_kws"]["edgecolor"] = {True:"white", False:"black"}[style == "dark_background"]
                            args_track = dict(
                                name=name,
                                divider=divider,
                                xlim=xlim,
                                ylim=ylim,
                                show_track_ticks=show_track_ticks,
                                show_track_labels=show_track_labels,
                                label_kws=_label_kws,
                            )
                            ax_track = self._plot_track_mpl( **args_track)
                            self.axes.append(ax_track)
                else:
                    print("Warning: `show_tracks` visualization is currently not available with polar projection", file=sys.stderr)
            # Add secondary classes
            if all([bool(show_secondary_classes), bool(self.secondary_classes)]):
                if not polar:
                    if show_secondary_classes == True:
                        show_secondary_classes = list(self.secondary_classes.keys())
                    assert set(show_secondary_classes) <= set(self.secondary_classes), "`show_secondary_classes` must be either `True`, `False`, or a subset of secondary classes that have been added manually"
                    for name, data in self.secondary_classes.items():
                        args_secondary_class = dict(
                            size=data["size"],
                            pad=data["pad"],
                            divider=divider,
                            xlim=xlim,
                            show_cluster_labels=False,
                            fill_kws=_fill_kws,
                            cluster_kws=_cluster_kws,
                            name=name,
                            track_label=name,
                            label_kws=_label_kws,
                            show_track_labels=show_track_labels,

                        )
                        ax_secondary_class = self._plot_classes_mpl( **args_secondary_class)
                        self.axes.append(ax_secondary_class)
                else:
                    print("Warning: `show_secondary_classes` visualization is currently not available with polar projection", file=sys.stderr)

            # Add clusters
            if all([show_clusters, self._clusters_exist]):
                if not polar:
                    args_clusters = dict(
                        size=f"{cluster_size}%",
                        pad=cluster_pad,
                        divider=divider,
                        xlim=xlim,
                        show_cluster_labels=show_cluster_labels,
                        fill_kws=_fill_kws,
                        cluster_kws=_cluster_kws,
                        name=None,
                        track_label=cluster_label,
                        label_kws=_label_kws,
                        show_track_labels=show_track_labels,
                    )
                    ax_clusters = self._plot_classes_mpl( **args_clusters)
                    self.axes.append(ax_clusters)
                else:
                    print("Warning: `show_clusters` visualization is currently not available with polar projection", file=sys.stderr)



            # Leaves
            if show_leaves:
                if leaf_colors is None:
                    if style == "dark_background":
                        leaf_colors = "white"
                    else:
                        leaf_colors = "black"
                if mpl.colors.is_color_like(leaf_colors):
                    leaf_colors = pd.Series([leaf_colors]*self.num_leaves, index=self.leaves)

                if len(self.axes) > 1:
                    [*map(lambda ax_query: ax_query.get_xaxis().set_visible(False), self.axes[:-1])]
                # Add leaf tick labels to polar projection plot
                if polar:
                    rmax = ax_dendrogram.get_rmax()
                    # Set pad
                    if pad is None:
                        rmin = ax_dendrogram.get_rmin()
                        pad = 0.01*rmin
                    # Add labels
                    for i, label in enumerate(self.axes[-1].get_xticklabels()):
                        leaf = self.leaves[i]
                        t = self.leaf_positions_polar[leaf]
                        label = self.axes[-1].text(t, rmax+pad, leaf, color=leaf_colors[leaf], transform=label.get_transform(), ha="center", va="center")
                        label.set_rotation(np.rad2deg(t))
                # Add leaf tick labels to rectilinear projection plot
                else:
                    self.axes[-1].set_xticks(self.leaf_ticks_rectilinear)
                    self.axes[-1].set_xticklabels(self.leaves, **_leaf_tick_kws)
                    self.axes[-1].xaxis.set_ticks_position("bottom")
                    for label in self.axes[-1].get_xticklabels():
                        label.set_color(leaf_colors[label.get_text()])
            else:
                self.axes[-1].set_xticklabels([])

            # Spines
            if polar:
                ax_dendrogram.spines["polar"].set_visible(show_spines)
            else:
                for spine_type, spine in ax_dendrogram.spines.items():
                    spine.set_visible(show_spines)

            # Legend
            if legend is not None:
                ax_dendrogram.legend(*format_mpl_legend_handles(legend), **_legend_kws)

            # Manage grids

            [*map(lambda ax_query: ax_query.xaxis.grid(show_xgrid), self.axes)]
            [*map(lambda ax_query: ax_query.yaxis.grid(show_ygrid), self.axes)]


            # Title
            if title is not None:
                fig.suptitle(title, **_title_kws)
            return fig, self.axes


    # ============
    # Analysis
    # ============
    def silhouette(self, cluster, func_distance=None, into=pd.Series):
        if self.tree_type == ete3.ClusterTree:
            subtree = self.get_subtrees(cluster)
            scores = subtree.get_silhouette(fdist=func_distance)
            labels = ["silhouette", "intracluster_dist", "intercluster_dist"]
            output = pd.Series(scores, index=labels, name=f"cluster_{cluster}")
            return into(output)
        else:
            print("Warning: `silhouette` score can only be calculated on `ete3.ClusterTree` objects", file=sys.stderr)

    # Eigengenes
    def eigengenes(self, X=None, mode=0):
        """
        Assumes X.shape follows (obsv_ids, attr_ids)
        mode == 0: return | eigengene pd.DataFrame
        mode == 1: return | eigenvalue pd.Series
        mode == 2: return | (0,1)
        """
        X_cluster = None
        if X is None:
            assert self.data is not None, "If X is None then `data` must be not be None during init"
            X = self.data
        # Updating leaf eaxis
        if self.leaf_axis is None:
            axis_0_overlap = len(set(self.linkage_labels) & set(X.index))
            axis_1_overlap = len(set(self.linkage_labels) & set(X.columns))
            leaf_axis = np.argmax([axis_0_overlap, axis_1_overlap])
            self.leaf_axis = leaf_axis
        assert self.leaf_axis in [0,1], "Could not infer leaf axis.  Please specify the leaf_axis when instantiating the model"

        d_cluster_eigengenes = dict()
        d_cluster_eigenvalues = dict()
        for cluster, leaves in self.get_clusters(grouped=True).iteritems():
            if self.leaf_axis == 0:
                X_cluster = X.loc[leaves,:].T
            if self.leaf_axis == 1:
                X_cluster = X.loc[:,leaves]
            self.pca_model = PrincipalComponentAnalysis(X_cluster, n_components=-1)
            d_cluster_eigengenes[cluster] = pd.Series(self.pca_model.projection_["PC.1"], index=X_cluster.index, name=cluster)
            d_cluster_eigenvalues[cluster] = self.pca_model.explained_variance_ratio_["PC.1"]

        if mode == 0:
            df_eigengenes = pd.DataFrame(d_cluster_eigengenes)
            df_eigengenes.columns.name = "cluster"
            return df_eigengenes
        if mode == 1:
            return pd.Series(d_cluster_eigenvalues)
        if mode == 2:
            return pd.DataFrame(d_cluster_eigengenes), pd.Series(d_cluster_eigenvalues)


    # Intramodular connectivity
    def intramodular_connectivity(self, kernel="auto"):
        if kernel == "auto":
            kernel = 1 - self.df_dism
        if is_query_class(kernel, "Symmetric"):
            kernel = kernel.to_dense()
        return intramodular_connectivity(kernel, clusters=self.get_clusters()[kernel.index])

    # ============
    # Conversion
    # ============
    def is_cluster_node(self, node):
        return node.name in self.cluster_attributes["node"].values

    def as_tree(self, format="ete3", collapse_clusters=False, prefix_cluster="cluster_"):
        accepted_formats = {"ete3", "skbio"}
        assert format in accepted_formats, f"Available formats must be one of the following: {accepted_formats}"
        if format == "ete3":
            if not collapse_clusters:
                return self.tree
            else:
                print("Warning: Collapsing clusters is experimental and should be checked manually before propogating downstream", file=sys.stderr)
                cluster_node = self.cluster_attributes["node"]
                node_cluster = pd.Series(cluster_node.index, index=cluster_node.values)

                if prefix_cluster is None:
                    prefix_cluster = ""

                newick = self.tree.write(is_leaf_fn=self.is_cluster_node, format=0)
                # tree_clusters = self.tree_type(newick)
                # name_tree_nodes(tree_clusters, node_prefix="y")
                tree_clusters = create_tree(newick=newick, into=self.tree_type, force_bifuraction=True)

                for node in tree_clusters.traverse():
                    if self.is_cluster_node(node):
                        id_cluster = node_cluster[node.name]
                        node.name = "{}{}".format(prefix_cluster, id_cluster)
                        for field in  self.cluster_attributes.columns:
                            node.add_feature(field, self.cluster_attributes.loc[id_cluster,field])

                return tree_clusters

        if format == "skbio":
            assert not collapse_clusters, "Cannot currently `collapse_clusters` with `skbio` trees.  Will be available in future versions."
            tree_skbio = skbio.TreeNode.from_linkage_matrix(self.Z, self.linkage_labels)
            tree_skbio.name = self.name
            return tree_skbio


    def as_graph(self, graph=None, weight_func=None):
        # Get graph
        if graph is None:
            graph = nx.OrderedDiGraph(name=self.name)

        # Construct graph
        if weight_func is not None:
            for node in self.tree.traverse():
                for child in node.get_children()[::-1]:
                    branch_length = self.tree.get_distance(node, child)
                    graph.add_edge(node.name, child.name, weight=weight_func(branch_length))
        else:
            for node in self.tree.traverse():
                for child in node.get_children()[::-1]:
                    branch_length = self.tree.get_distance(node, child)
                    graph.add_edge(node.name, child.name, weight=branch_length)
        # Add clusters
        if hasattr(self, "clusters"):
            for (leaf, cluster_data) in self.leaf_attributes.iterrows():
                graph.node[leaf].update(cluster_data.to_dict())
        return graph


# Hierarchical Topology
class Topology(object):
    def __init__(self, X=None, y=None, name=None, class_type=None, attr_type=None, obsv_type=None, metric="euclidean", linkage_method="ward", class_colors="gray", metadata=dict(), initialize=True, axis=0, verbose=True, label_explained_variance_ratio="EV Ratio"):
        self.X = X
        self.y = y
        self.name = name
        self.metric = metric
        self.linkage_method = linkage_method
        self.axis = axis
        self.class_type = class_type
        self.attr_type = attr_type
        self.obsv_type = obsv_type
        self.verbose = verbose
        self.tracks = OrderedDict()
        self.class_colors = class_colors
        self.classes = sorted(self.y.unique())
        self.label_explained_variance_ratio = label_explained_variance_ratio
        self.__synthesized__ = datetime.datetime.utcnow()
        self.metadata = {
            "name":name,
            "metric":metric,
            "linkage_method":linkage_method,
            "dimensions":X.shape if X is not None else None,
            "class_type":class_type,
            "classes":self.classes,
            "attr_type":attr_type,
            "obsv_type":obsv_type,
            "axis":axis,
            "verbose":verbose,
            "synthesized":self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.metadata.update(metadata)
        if initialize:
            condition_1 = X is not None
            condition_2 = y is not None
            if all([condition_1, condition_2]):
                self.compute(overwrite=True)
            else:
                print("X and y must be given to run computation. Returning uninitialized object", file=sys.stderr)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.metadata)[1:-1]})"
    def copy(self):
        return copy.deepcopy(self)

    def _check_proceed(self, attribute, overwrite):
        proceed = True
        if hasattr(self, attribute):
            condition_1 = getattr(self, attribute) is not None
            condition_2 = not overwrite
            if all([condition_1, condition_2]):
                proceed = False
        return proceed

    def eigendecomposition(self, overwrite=False):
        assert self.X is not None, "`X` can not be None to do `eigendecomposition`"
        proceed = self._check_proceed(attribute="eigenprofiles", overwrite=overwrite)
        if proceed:
            self.eigenprofiles, self.eigenvalues = eigenprofiles_from_data(self.X, y=self.y, class_type=self.class_type, mode=None)
        return self

    def dissimilarity(self, overwrite=False):
        assert hasattr(self, "eigenprofiles"), "Must compute `eigendecomposition` before calculating `dissimilarity`"
        proceed = self._check_proceed(attribute="dism", overwrite=overwrite)
        if proceed:
            self.dism = pairwise(self.eigenprofiles, metric=self.metric, axis=0)
            self.labels = self.dism.index
        return self

    def cluster(self, overwrite=False):
        assert hasattr(self, "dism"), "Must compute `dissimilarity` before calculating `cluster`"
        proceed = self._check_proceed(attribute="Z", overwrite=overwrite)
        if proceed:
            self.Z = dism_to_linkage(self.dism, method=self.linkage_method)
        return self

    def dendrogram(self, overwrite=False):
        assert hasattr(self, "Z"), "Must compute `cluster` before calculating `dendrogram`"
        proceed = self._check_proceed(attribute="leaves", overwrite=overwrite)
        if proceed:
            dendrogram_data = sp_hierarchy.dendrogram(self.Z, no_plot=True, labels=self.labels)
            self.leaves = dendrogram_data["ivl"]
            self.icoord = np.asarray(dendrogram_data["icoord"])
            self.dcoord = np.asarray(dendrogram_data["dcoord"])
            self.num_leaves = len(dendrogram_data["ivl"])
            self.leaf_positions = pd.Series(np.arange(5, self.num_leaves * 10 + 5, 10), index=self.leaves)
            self.add_track(name=self.label_explained_variance_ratio, data=self.eigenvalues, color=self.class_colors, plot_kws={"linewidth":1.382})
            self.add_track(name="Size", data=self.y.value_counts(), color=self.class_colors, plot_kws={"linewidth":1.382})
        return self
    def convert_to_tree(self, node_prefix="y", overwrite=False):
        assert hasattr(self, "leaves"), "Must compute `dendrogram` before converting to `ete3.Tree` object`"
        proceed = self._check_proceed(attribute="tree", overwrite=overwrite)
        if proceed:
            newick = linkage_to_newick(self.Z, self.labels)
            root_node = f"{node_prefix}1"
            self.tree = ete3.Tree(newick=newick, name=root_node)
            node_index = int(root_node[-1]) + 1
            self.submodels = [root_node]
            for node in self.tree.iter_descendants():
                if not node.is_leaf():
                    node.name = f"{node_prefix}{node_index}"
                    self.submodels.append(node.name)
                    node_index += 1
            self.internal_node_positions = self._get_internal_nodes_positions(self.icoord, self.dcoord, self.leaf_positions, self.tree)
        return self
    def convert_to_graph(self, overwrite=False):
        # Could use `ete_to_nx` here instead
        assert hasattr(self, "tree"), "Must compute `tree` before converting to `nx.OrderedDiGraph` object`"
        proceed = self._check_proceed(attribute="graph", overwrite=overwrite)
        if proceed:
            self.graph = nx.OrderedDiGraph(name=self.name)
            for node in self.tree.traverse():
                for child in node.get_children()[::-1]:
                    self.graph.add_edge(node.name, child.name)
        return self


    def compute(self, overwrite=True):
        start_time = time.time()
        if self.verbose:
            print("===============================", file=sys.stdout)
            print("Running computational pipeline:", file=sys.stdout)
            print("===============================", file=sys.stdout)
        for computation_step in ["eigendecomposition", "dissimilarity", "cluster", "dendrogram", "convert_to_tree", "convert_to_graph"]:
            if self.verbose: print(computation_step, format_duration(start_time), sep="\t", file=sys.stdout)
            getattr(self,computation_step)(overwrite=overwrite)
        return self
    # ==========
    # Conversion
    # ==========

    # Relabel the submodel nodes
    def relabel_submodels(self, mapping):
        assert isinstance(mapping, Mapping), "`mapping` must be a type of dictionary"
        assert hasattr(self, "tree"), "Must create tree first"


        if set(mapping.values()) >= set(self.submodels):
            if self.verbose:
                print("Skipping relabeling because this dictionary has already been used to relabel the original node/submodel names", file=sys.stderr)
        else:
            assert set(mapping) >= set(self.submodels), f"`mapping` must have a key:value pair for each submodel node: {self.submodels}"
            for i, submodel_node in enumerate(self.submodels):
                node = self.tree.search_nodes(name=submodel_node)[0]
                node.name = self.submodels[i] = mapping[submodel_node]
            if hasattr(self, "graph"):
                self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        return self

    # =======
    # Get data
    # =======
    # Get text representation of tree
    def get_ascii(self, show_internal=True, compact=False):
        """
        Get the ascii text tree
        """
        attributes=None
        assert hasattr(self, "tree"), "Must create tree first"
        return self.tree.get_ascii(show_internal=show_internal, compact=compact, attributes=attributes)

    # Get target paths
    def get_paths(self, target=None):
        assert hasattr(self,"graph"), "Must create graph first"

        root = self.tree.name
        if target is None:
            paths = list()
            for target in self.classes:
                paths.append(nx.shortest_path(self.graph, source=root, target=target))
            return paths
        else:
            assert target in self.classes, f"{target} is not in `self.classes`"
            return nx.shortest_path(self.graph, source=root, target=target)

    # Get target matrix
    def get_target_matrix(self, directory=None):
        if directory is not None:
            if type(directory) == str:
                directory = pathlib.Path(directory)
            os.makedirs(directory, exist_ok=True)
        # Nested
        def _get_next_node(current_node, target, descendant_classes):
            path = nx.shortest_path(self.graph, source=current_node, target=target)
            next_node = path[1]
            return next_node
        # Y target matrix
        Y = dict()
        # Iterate through submodel nodes
        for submodel in self.submodels:
            # Get the node object
            node = self.tree.search_nodes(name=submodel)[0]
            # Get the child nodes
            # Get the classes that are lower in the hierarchy
            descendant_classes = node.get_leaf_names()
            # Get the training observations that are in the hierarchy
            y_mask = self.y.map(lambda x: x in descendant_classes)
            y_subset = self.y[y_mask]
            # Path
            y_next = y_subset.map(lambda target:_get_next_node(node.name, target, descendant_classes))
            Y[submodel] = y_next
            # Write training files to a directory
            if directory is not None:
                directory_submodel = os.path.join(directory,submodel)
                os.makedirs(directory_submodel, exist_ok=True)
                self.X.loc[y_next.index,:].to_pickle(os.path.join(directory_submodel, "X.pbz2"), compression="bz2")
                self.y.to_frame(self.class_type).loc[y_next.index,:].to_pickle(os.path.join(directory_submodel, "y.pbz2"), compression="bz2")
        Y = pd.DataFrame(Y).loc[self.X.index,:]
        return Y

    def _get_internal_nodes_positions(self, icoord,dcoord,leaf_positions, tree):
        # Get x positions for each leaf
        leaf_positions = leaf_positions.to_dict()
        x_position_leaf = {v:k for k,v in leaf_positions.items()}
        # Sort in ascending order of height so we are working our way up from the leaves to the root
        idx_ascending_tree = np.argsort(dcoord[:,1])

        # Dynamically set the internal node names as we work our way up so the internal nodes are treated as leaves
        internal_node_positions = dict()
        for i_data, d_data in zip(icoord[idx_ascending_tree], dcoord[idx_ascending_tree]):
            leaf_positions_in_branch = set(i_data) & set(x_position_leaf.keys())
            leaves = [*map(lambda x: x_position_leaf[x], leaf_positions_in_branch)]
            node_name = tree.get_common_ancestor(leaves).name
            position = i_data
            x_position_leaf[np.mean(i_data)] = node_name
            internal_node_positions[node_name] = (np.mean(i_data), np.max(d_data))
        return internal_node_positions

    # ======
    # Tracks
    # ======
    def add_track(self, name, data, alpha=0.618, color="black", pad=0.25, percent_of_figure = 20, ylim=None, plot_type="bar", spines=["top", "bottom", "left", "right"], fillna=0, plot_kws=dict(), label_kws=dict(),  bar_width="auto", check_missing_values=False, convert_to_hex=False, horizontal_kws=dict(), horizontal_lines=[]):
        """
        type(track) should be a pd.Series
        """


        # Keywords
        if plot_type == "bar":
            _plot_kws = {"linewidth":1.382, "alpha":alpha, "width":max(5,self.num_leaves/30) if bar_width == "auto" else bar_width}
        elif plot_type == "area":
            _plot_kws = {"linewidth":1.382, "alpha":alpha}
        else:
            _plot_kws = {}
        _plot_kws.update(plot_kws)

        _horizontal_kws = {"linewidth":1, "linestyle":"-", "color":"black"}
        _horizontal_kws.update(horizontal_kws)

        _label_kws = {"fontsize":15, "rotation":0, "horizontalalignment":"right"}
        _label_kws.update(label_kws)

        # Color of the track
        if color is None:
            color = "black"
        if isinstance(color,str):
            color = pd.Series([color]*len(self.leaves), index=self.leaves)
        if is_dict(color):
            color = pd.Series(color)
        if convert_to_hex:
            _sampleitem_ = color[0]
            assert mpl.colors.is_color_like(_sampleitem_), f"{_sampleitem_} is not color-like"
            if is_rgb_like(_sampleitem_):
                color = color.map(rgb2hex)
        # Check for missing values
        if check_missing_values:
            assert set(data.index) == set(self.leaves), "`data.index` is not the same set as `self.leaves`"
            assert set(color.index) == set(self.leaves), "`color.index` is not the same set as `self.leaves`"
        if plot_type in ["area","line"]:
            if color.nunique() > 1:
                print("Warning: `area` and `line` plots can only be one color.  Defaults to first color in iterable.", file=sys.stderr)
        # Horizontal lines
        if is_number(horizontal_lines):
            horizontal_lines = [horizontal_lines]

        # Convert to ticks
        data = data[self.leaves]
        data.index = data.index.map(lambda x:self.leaf_positions[x])
        color = color[self.leaves]
        color.index = color.index.map(lambda x:self.leaf_positions[x])
        # Store data
        self.tracks[name] = {"data":data, "color":color, "pad":pad, "size":f"{percent_of_figure}%", "plot_type":plot_type, "plot_kws":_plot_kws, "label_kws":_label_kws, "spines":spines, "ylim":ylim, "horizontal_kws":_horizontal_kws, "horizontal_lines":horizontal_lines}

        return self
    # ============
    # Plotting
    # ============
    def _plot_dendrogram_mpl(self, icoord, dcoord, ivl, xlim, ylim, color, ax, distance_label, fig_kws, line_kws, distance_tick_kws):
        # Plot dendrogram
        for xs, ys in zip(icoord, dcoord):
            ax.plot(xs, ys,  color, **line_kws)
        # Adjust ticks
        ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        #Distance label
        if distance_label is not None:
            ax.set_ylabel(distance_label, **distance_tick_kws)
        # Append axes objects
        self.axes.append( ax )

    def _plot_nodes_mpl(self, icoord, dcoord, ax, pad,annot_kws=dict()):



        ylim = ax.get_ylim()

        for name, (x,y) in self.internal_node_positions.items():
            plt.plot(x, y, "")
            plt.annotate(name, (x, y),
                         **annot_kws)
        ax.set_ylim(min(ylim), max(ylim)+0.0382*pad)
        return ax


    def _plot_track_mpl(self, name, divider, xlim, ylim, show_track_ticks, show_track_labels):
        ax = divider.append_axes("bottom", size=self.tracks[name]["size"], pad=self.tracks[name]["pad"])
        data = self.tracks[name]["data"]
        c = self.tracks[name]["color"]
        kws = self.tracks[name]["plot_kws"]


        if self.tracks[name]["plot_type"] == "bar":
            ax.bar(data.index, data.values, color=c, **kws)
            kws_outline = kws.copy()
            del kws_outline["alpha"]
            ax.bar(data.index, data.values, color="none", edgecolor=c, **kws_outline)
        else:
            if self.tracks[name]["plot_type"] == "area":
                data.plot(kind="area", color=c.values, ax=ax, **kws)
                data.plot(kind="line", color=c.values, alpha=1,  ax=ax, **kws)
            else:
                data.plot(kind=self.tracks[name]["plot_type"], color=c.values, ax=ax,  **kws)

        if len(self.tracks[name]["horizontal_lines"]):
            for y_pos in self.tracks[name]["horizontal_lines"]:
                ax.axhline(y_pos, **self.tracks[name]["horizontal_kws"])
        # Spines
        for spine in ["top", "bottom", "left", "right"]:
            if spine not in self.tracks[name]["spines"]:
                ax.spines[spine].set_visible(False)
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        if self.tracks[name]["ylim"] is not None:
            ax.set_ylim(ylim)
        if not show_track_ticks:
            ax.set_yticklabels([])
        if show_track_labels:
            ax.set_ylabel(name, **self.tracks[name]["label_kws"])
        return ax


    # Master plot method
    def plot_topology(self,
             branch_color="auto",
             leaf_font_size=15,
             leaf_rotation=0,
             style="seaborn-white",
             figsize=(13,5),
             title=None,
             ax=None,
             legend=False,
             fig_kws=dict(),
             line_kws=dict(),
             leaf_tick_kws=dict(),
             distance_tick_kws=dict(),
             distance_label="Distance",
             title_kws=dict(),
             title_position=0.95,
             show_class_size=False,
             show_explained_variance_ratio=True,
             annot_kws=dict(),
             # bar_kws=dict(),
            box_facecolor="whitesmoke",
              box_edgecolor="black",
                      box_style="round",
                      box_linewidth=1,
                      box_pad=0.25,
                      box_alpha=0.85,
            ):

        # Initialize
        self.axes = list()
        orientation = "top" # Does not work with bottom, left, or right yet 2018-June-26

        # Limits
                # Independent variable plot width
        tree_width = len(self.leaves) * 10
        xlim = (0,tree_width)
            # Depenendent variable plot height
        maximum_height = max(self.Z[:, 2])
        tree_height = maximum_height + maximum_height * 0.05
        ylim = (0,tree_height)

        # Keywords
        _fig_kws = {"figsize":figsize}
        _fig_kws.update(fig_kws)
        _line_kws = {"linestyle":"-", "linewidth":1.382}
        _line_kws.update(line_kws)
        # _bar_kws = {"linewidth":1.382}
        # _bar_kws.update(bar_kws)
        _distance_tick_kws = {"fontsize":15, "rotation":90}
        if _distance_tick_kws["rotation"] == 0:
            _distance_tick_kws["horizontalalignment"] = "right"
        _distance_tick_kws.update(distance_tick_kws)
        _leaf_tick_kws = {"fontsize":15, "rotation":leaf_rotation}
        _leaf_tick_kws.update(leaf_tick_kws)
        _title_kws = {"fontsize":18, "fontweight":"bold","y":title_position}
        _title_kws.update(title_kws)
        _annot_kws = dict(fontsize=15, fontweight="bold", xytext=(10,15), textcoords='offset points', verticalalignment='top', horizontalalignment='center', bbox= dict(boxstyle=box_style, fc=box_facecolor, ec=box_edgecolor, linewidth=box_linewidth,  alpha=box_alpha))
        _annot_kws.update(annot_kws)
        # Branch colors
        if branch_color == "auto":
            branch_color = {True:"white", False:"black"}[style == "dark_background"]


        # Plotting
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(**_fig_kws)
            else:
                fig = plt.gcf()

            # Arguments for dendrogram plotting
            args_dendro = dict(
                icoord=self.icoord,
                dcoord=self.dcoord,
                ivl=self.leaves,
                color=branch_color,
                xlim=xlim,
                ylim=ylim,
                ax=ax,
                fig_kws=_fig_kws,
                line_kws=_line_kws,
                distance_tick_kws=_distance_tick_kws,
                distance_label=distance_label,
            )

            self._plot_dendrogram_mpl(**args_dendro)
            self._plot_nodes_mpl(icoord=self.icoord, dcoord=self.dcoord, ax=ax, pad=box_pad,annot_kws=_annot_kws)

            # Add tracks
            divider = make_axes_locatable(ax)
            additional_tracks = list()
            if show_explained_variance_ratio:
                additional_tracks.append(self.label_explained_variance_ratio)
            if show_class_size:
                additional_tracks.append("Size")

            if len(additional_tracks) > 0:
                for name in additional_tracks:
                    if "edgecolor" in self.tracks[name]["plot_kws"]:
                        if self.tracks[name]["plot_kws"]["edgecolor"] == "auto":
                            self.tracks[name]["plot_kws"]["edgecolor"] = {True:"white", False:"black"}[style == "dark_background"]
                    args_track = dict(
                        name=name,
                        divider=divider,
                        xlim=xlim,
                        ylim=ylim,
                        show_track_ticks=False,
                        show_track_labels=True,
                    )
                    ax_track = self._plot_track_mpl( **args_track)
                    self.axes.append(ax_track)

            # Leaves
            [*map(lambda ax_query: ax_query.get_xaxis().set_visible(False), self.axes[:-1])]
            self.axes[-1].set_xticks(self.leaf_positions.values)
            self.axes[-1].set_xticklabels(self.leaves, **_leaf_tick_kws)
            self.axes[-1].xaxis.set_ticks_position("bottom")

            # Remove grids
            [*map(lambda ax_query: ax_query.grid(False), self.axes)]

            # Title
            if title is None:
                title = self.name
            if title is not None:
                fig.suptitle(title, **_title_kws)
            return fig, ax
