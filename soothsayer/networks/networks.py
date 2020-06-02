
# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, itertools, copy, warnings, datetime
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import interp1d
from scipy import stats
from scipy.spatial.distance import squareform
# from teneto import TemporalNetwork as TenetoTemporalNetwork
from tqdm import tqdm
from skbio.util._decorator import experimental, stable

# Soothsayer
from ..symmetry import *
from ..utils import *
# from ..r_wrappers.packages.WGCNA import pickSoftThreshold_fromSimilarity
from ..visuals import plot_scatter, bezier_points
from ..transmute.normalization import normalize_minmax
from ..io import write_object



__all__ = ["Hive", "intramodular_connectivity", "topological_overlap_measure", "signed", "determine_soft_threshold","cluster_modularity", "TemporalNetwork", "Edge"]
__all__ = sorted(__all__)

# Network Edge
@experimental(as_of="2019.08")
class Edge(tuple):

    """
    # ------------
    # Create edges
    # ------------
    edge_1 = Edge(["Anakin", "Darth Vader"], weight=-0.618, directed=False)
    edge_2 = Edge(["Darth Sidious", "Darth Vader"], weight=0.8, directed=True, affiliation="Sith")
    edge_3 = Edge(["Darth Plagueis", "Darth Plagueis"], weight=1, affiliation="Sith")

    # --------------
    # Representation
    # --------------

    print(edge_1)
    # Edge(Anakin --- Darth Vader, w=-0.61800)

    print(edge_2)
    # Edge(Darth Sidious --> Darth Vader, w=0.80000)

    print(edge_3)
    # Edge(Darth Plagueis === Darth Plagueis, w=1.00000)

    # --------
    # Metadata
    # --------
    print(edge_2.metadata)
    # {'affiliation': 'Sith'}

    # -----------------
    # Set comprehension
    # -----------------
    edge_1 & edge_2
    # frozenset({'Darth Vader'})

    print(edge_1 | edge_2 | edge_3)
    # frozenset({'Anakin', 'Darth Plagueis', 'Darth Sidious', 'Darth Vader'})

    print(edge_1 < {"Anakin", "Darth Vader", "Darth Tenebrous"})
    # True

    """
    # Instantiate
    def __new__(self, edge, weight=1, directed=False, **metadata):
        return super(Edge, self).__new__(self,tuple(edge))

    def __init__(self, edge, weight=1, directed=False, **metadata):
        # Store metadata
        self.metadata = metadata
        # Self-loops
        self.self_loop = False
        if len(set(edge)) == 1:
            edge = [edge[0], edge[0]]
            directed = None
            self.self_loop = True
        # Edge
        self.directed = directed
        if self.directed:
            self.edge = tuple(self)
        else:
            self.edge = tuple(sorted(self))
        self.sign = np.sign(weight)
        self.weight = abs(weight)
        # Nodes
        self.nodes = frozenset(edge)
        # Dtypes
        self.dtypes = dict(zip(self, (map(type, self))))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.edge[key]
        else:
            return self.metadata[key]

    def __repr__(self):
        return "Edge(%s %s %s, w=%0.5f)"%(self.edge[0], {True:"-->", False:"---", None:"==="}[self.directed], self.edge[1], self.weight*self.sign)

    # Override set specials
    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        iterable = frozenset(other)
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        other = frozenset(other)
        return self >= other and self != other
    def __and__(self, other):
        other = frozenset(other)
        return self.nodes.intersection(other)
    def __or__(self, other):
        other = frozenset(other)
        return self.nodes.union(other)

    def __rand__(self, other):
        other = frozenset(other)
        return other.intersection(self.nodes)
    def __ror__(self, other):
        other = frozenset(other)
        return other.union(self.nodes)
# =======================================================
# Hive
# =======================================================
class Hive(object):
    def __init__(self, kernel,  name=None,  node_type=None, axis_type=None, metric_type=None, description=None, verbose=False):
        """
        Hive plots:
        Should only be used with 2-3 axis unless intelligently ordered b/c the arcs will overlap.

        Usage:
        # Adjacency matrix
        # ================
        df_adj = d_phenotype_adjacency["Diseased"]["species"] # <- (n,n) pd.DataFrame

        # Organize the axis nodes
        # =======================
        ## Axis 1
        idx_streptococcus = set([*filter(lambda id_species: id_species.split(" ")[0] == "Streptococcus", df_adj.index)])
        ## Axis 2
        idx_veillonella = set([*filter(lambda id_species: id_species.split(" ")[0] == "Veillonella", df_adj.index)])
        ## Axis 3
        idx_actinomyces = set([*filter(lambda id_species: id_species.split(" ")[0] == "Actinomyces", df_adj.index)])
        ## All nodes in hive
        idx_nodes = set(idx_streptococcus | idx_veillonella | idx_actinomyces)

        # Create the Hive
        # ===============
        # Instantiate the hive
        hive = Hive(df_adj.loc[idx_nodes,idx_nodes], verbose=False, name="Diseased")
        # Add the axes
        hive.add_axis("$Streptococcus$", nodes=idx_streptococcus, colors="magenta", split_axis=False)
        hive.add_axis("$Veillonella$", nodes=idx_veillonella, colors="red", split_axis=True)
        hive.add_axis("$Actinomyces$", nodes=idx_actinomyces, colors="orange", split_axis=True)
        # Compile to get the angles for all of the axes
        hive.compile()
        # Plot
        fig, ax = hive.plot(title="Diseased",  style="seaborn-white", ax=axes[i], axis_label_kws={"fontsize":15}, title_kws={"fontsize":18})
        """
        # Initialise

        if isinstance(kernel, pd.DataFrame):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kernel = Symmetric(kernel, data_type=node_type, metric_type=metric_type, name=name, mode='similarity', force_the_symmetry=True)
        assert is_query_class(kernel, "Symmetric"), "`kernel` should either be a Symmetric object or a symmetric pd.DataFrame of adjacencies"
        if not np.all(kernel.data.values >= 0):
            warnings.warn("Adjacency weights must be ≥ 0 to plot.  The signs are preserve in the Hive but absolute values will be computed for weighted edge plotting.")

        self.kernel = kernel
        self.name = name
        self.verbose = verbose
        self.axes = OrderedDict()
        self.node_mapping_ = OrderedDict()
        self.compiled = False
        self.node_type = node_type
        self.axis_type = axis_type
        self.metric_type = metric_type
        self.description = description
        self.__synthesized__ = datetime.datetime.utcnow()
        self.number_of_nodes_ = None
        if self.verbose:
            class_name = str(self.__class__).split(".")[-1][:-2]
            lines = [f"Hive| {self.name}"]
            lines.append(len(lines[0])*"=")
            lines.append("\n".join([
                f"      node_type: {self.node_type}",
                f"      axis_type: {self.axis_type}",
                f"      metric_type: {self.metric_type}",
                f"      description: {self.description}",
            ]))
            lines.append("      " + self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S"))
            print( "\n".join(lines), file=sys.stderr)
            print("-"*44, file=sys.stderr)


    # Built-ins
    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1][:-2]
        lines = [f"Hive| {self.name}"]
        lines.append(len(lines[0])*"=")
        lines.append("\n".join([
            f"      node_type: {self.node_type}",
            f"      axis_type: {self.axis_type}",
            f"      metric_type: {self.metric_type}",
            f"      description: {self.description}",
        ]))
        lines.append("\n".join([
            f"      number_of_axes: {len(self.axes)}",
            f"      number_of_nodes: {self.number_of_nodes_}",
        ]))
        lines.append("      " + self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S"))
        return "\n".join(lines)

    def __call__(self, name_axis=None):
        return self.get_axis_data(name_axis=name_axis)
    def __getitem__(self, key):
        return self.kernel[key]

    # Add axis to HivePlot
    def add_axis(self, name_axis, nodes, sizes=None, colors=None, split_axis:bool=False, node_style="o", scatter_kws=dict()):
        """
        Add or update axis
        """
        # Initialize axis container
        if self.verbose:
            if name_axis not in self.axes:
                print("      Adding axis:", name_axis, sep="\t", file=sys.stderr)
            else:
                print("      Updating axis:", name_axis, sep="\t", file=sys.stderr)
        self.axes[name_axis] = defaultdict(dict)
        self.axes[name_axis]["colors"] = None
        self.axes[name_axis]["sizes"] = None
        self.axes[name_axis]["split_axis"] = split_axis
        self.axes[name_axis]["node_style"] = node_style
        self.axes[name_axis]["scatter_kws"] = scatter_kws

        # Assign (preliminary) node positions
        if is_nonstring_iterable(nodes) and not isinstance(nodes, pd.Series):
            nodes = pd.Series(np.arange(len(nodes)), index=nodes)
        if is_dict(nodes):
            nodes = pd.Series(nodes)
        nodes = nodes.sort_values()
        assert set(nodes.index) <= set(self.kernel.labels), "All nodes in axis should be in the kernel and they aren't..."
        self.axes[name_axis]["node_positions"] = nodes
        self.axes[name_axis]["nodes"] = nodes.index
        self.axes[name_axis]["num_nodes"] = nodes.size

        # Group node with axis
        self.node_mapping_.update(dict_build([(name_axis, self.axes[name_axis]["nodes"])]))

        # Assign component colors
        if colors is None:
            colors = "white"
        if is_color(colors):
            colors = dict_build([(colors, self.axes[name_axis]["nodes"])])
        if is_dict(colors):
            colors = pd.Series(colors)
        if not is_color(colors):
            if is_nonstring_iterable(colors) and not isinstance(colors, pd.Series):
                colors = pd.Series(colors, index=self.axes[name_axis]["nodes"])
        self.axes[name_axis]["colors"] = colors[self.axes[name_axis]["nodes"]]

        # Assign component sizes
        if sizes is None:
            sizes = 100
        if is_number(sizes):
            sizes = dict_build([(sizes, self.axes[name_axis]["nodes"])])
        if is_dict(sizes):
            sizes = pd.Series(sizes)
        self.axes[name_axis]["sizes"] = sizes[nodes.index]

    # Compile the data for plotting
    def compile(self, split_theta_degree=None, inner_radius=None, theta_center=90, axis_normalize=True, axis_maximum=1000):
        """
        inner_radius should be similar units to axis_maximum
        """
        n_axis = len(self.axes)
        if split_theta_degree is None:
            split_theta_degree = (360/n_axis)*0.16180339887
        self.split_theta_degree = split_theta_degree
        if inner_radius is None:
            if axis_normalize:
                inner_radius = (1/5)*axis_maximum
            else:
                inner_radius = 3
        self.inner_radius = inner_radius
        self.theta_center = theta_center
        if self.verbose:
            print("-"*44, file=sys.stderr)
            print("      number of axes:", n_axis, sep="\t", file=sys.stderr)
            print("      inner_radius:", inner_radius, sep="\t", file=sys.stderr)
            print("      split_theta_degree:", split_theta_degree, sep="\t", file=sys.stderr)
            print("      axis_maximum:", axis_maximum, sep="\t", file=sys.stderr)
        # Adjust all of the node_positions
        for i, query_axis in enumerate(self.axes):
            # If the axis is normalized, force everything between the minimum position and the `axis_maximum`
            if axis_normalize:
                node_positions = self.axes[query_axis]["node_positions"]
                self.axes[query_axis]["node_positions_normalized"] = normalize_minmax(node_positions, feature_range=(min(node_positions), axis_maximum) )
            else:
                self.axes[query_axis]["node_positions_normalized"] = self.axes[query_axis]["node_positions"].copy()
            # Offset the node positions by the inner radius
            self.axes[query_axis]["node_positions_normalized"] = self.axes[query_axis]["node_positions_normalized"] + self.inner_radius

        # Adjust all of the axes angles
        for i, query_axis in enumerate(self.axes):
            # If the axis is in single mode
            if not self.axes[query_axis]["split_axis"]:
                # If the query axis is the first then the `theta_add` will be 0
                theta_add = (360/n_axis)*i
                self.axes[query_axis]["theta"] = np.array([self.theta_center + theta_add])
            else:
                theta_add = (360/n_axis)*i
                self.axes[query_axis]["theta"] = np.array([self.theta_center + theta_add - split_theta_degree,
                                                           self.theta_center + theta_add + split_theta_degree])
            self.axes[query_axis]["theta"] = np.deg2rad(self.axes[query_axis]["theta"])

        # Nodes
        # Old method: self.nodes_ = flatten([*map(lambda axes_data:axes_data["nodes"], self.axes.values())])
        # Not using old method because frozenset objects and tuples are unfolded.
        self.nodes_ = list()
        for axes_data in self.axes.values():
            self.nodes_ += list(axes_data["nodes"])

        assert len(self.nodes_) == len(set(self.nodes_)), "Axes cannot contain duplicate nodes"
        self.number_of_nodes = len(self.nodes_)

        with warnings.catch_warnings(): # patch until I fix the diagonal update in Symmetric.
            warnings.simplefilter("ignore")
            self.kernel_subset_ = Symmetric(self.kernel.to_dense().loc[self.nodes_, self.nodes_], data_type=self.node_type, metric_type=self.metric_type, name=self.name, mode='similarity', force_the_symmetry=True) # Need a more efficient method
        # Compile
        self.compiled = True

    def _get_quadrant_info(self, theta_representative):
        # 0/360
        if theta_representative == np.deg2rad(0):
            horizontalalignment = "left"
            verticalalignment = "center"
            quadrant = 0
        # 90
        if theta_representative == np.deg2rad(90):
            horizontalalignment = "center"
            verticalalignment = "bottom"
            quadrant = 90
        # 180
        if theta_representative == np.deg2rad(180):
            horizontalalignment = "right"
            verticalalignment = "center"
            quadrant = 180
        # 270
        if theta_representative == np.deg2rad(270):
            horizontalalignment = "center"
            verticalalignment = "top"
            quadrant = 270

        # Quadrant 1
        if np.deg2rad(0) < theta_representative < np.deg2rad(90):
            horizontalalignment = "left"
            verticalalignment = "bottom"
            quadrant = 1
        # Quadrant 2
        if np.deg2rad(90) < theta_representative < np.deg2rad(180):
            horizontalalignment = "right"
            verticalalignment = "bottom"
            quadrant = 2
        # Quadrant 3
        if np.deg2rad(180) < theta_representative < np.deg2rad(270):
            horizontalalignment = "right"
            verticalalignment = "top"
            quadrant = 3
        # Quadrant 4
        if np.deg2rad(270) < theta_representative < np.deg2rad(360):
            horizontalalignment = "left"
            verticalalignment = "top"
            quadrant = 4
        return quadrant, horizontalalignment, verticalalignment

    def plot(self,
             title=None,
             # Arc style
             arc_style="curved",
             # Show components
             show_axis=True,
             show_edges=True,
             show_border = False,
             show_axis_labels=True,
             show_node_labels=False,
             show_grid=False,
             # Colors
             axis_color=None,
             edge_colors=None,
             background_color=None,
             # Alphas
             edge_alpha=0.5,
             node_alpha=0.75,
             axis_alpha=0.618,
             # Keywords
             title_kws=dict(),
             axis_kws=dict(),
             axis_label_kws=dict(),
             node_label_kws=dict(),
             node_kws=dict(),
             edge_kws=dict(),
             legend_kws=dict(),
             # Figure
             style="dark",
             edge_linestyle="-",
             axis_linestyle="-",
             legend=None,
             polar=True,
             ax=None,
             clip_edgeweight=5,
             granularity=100,
             func_edgeweight=None,
             figsize=(10,10),
             # Padding
             pad_axis_label = "infer",
             pad_node_label = 65,
             node_pad_fill = "·", # default is an interpunct not a period/full-stop
             node_label_position_vertical_axis="right",
            ):
        assert self.compiled == True, "Please `compile` before plotting"
        accepted_arc_styles = {"curved", "linear"}
        assert arc_style in accepted_arc_styles, f"`arc_style` must be one of the following: {accepted_arc_styles}"
        if arc_style == "linear":
            granularity = 2
        if style in ["dark",  "black", "night",  "sith", "sun"]:
            style = "dark_background"
        if style in ["light", "white", "day", "jedi", "moon"] :
            style = "seaborn-white"

        with plt.style.context(style):
            # Create figure
            if ax is not None:
                fig = plt.gcf()
                figsize = fig.get_size_inches()
            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = plt.subplot(111, polar=polar)
            if polar == True:
                y = 1.05
            if polar == False:
                y = 1.1

            # Remove clutter from plot
            ax.grid(show_grid)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if not show_border: # Not using ax.axis('off') becuase it removes facecolor
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Default colors
            if axis_color is None:
                if style == "dark_background":
                    axis_color = "white"
                else:
                    axis_color = "darkslategray"
            if background_color is not None:
                ax.set_facecolor(background_color)

            # Title
            _title_kws = {"fontweight":"bold"}
            _title_kws.update(title_kws)
            if "fontsize" not in _title_kws:
                _title_kws["fontsize"] = figsize[0] * np.sqrt(figsize[0])/2 + 2
            # Axis labels
            _axis_label_kws = {"fontweight":"bold"}
            _axis_label_kws.update(axis_label_kws)
            if "fontsize" not in _axis_label_kws:
                _axis_label_kws["fontsize"] = figsize[0] * np.sqrt(figsize[0])/2
            # Node labels
            _node_label_kws = {"fontsize":10}
            _node_label_kws.update(node_label_kws)
            # Axis plotting
            _axis_kws = {"linewidth":3.382, "zorder":0, "alpha":axis_alpha, "color":axis_color, "linestyle":axis_linestyle}
            _axis_kws.update(axis_kws)
            # Node plotting
            _node_kws = {"linewidth":1.618, "edgecolor":axis_color, "alpha":node_alpha, "zorder":1}
            _node_kws.update(node_kws)
            # Edge plotting
            _edge_kws = {"alpha":edge_alpha, "zorder":_node_kws["zorder"]+2, "linestyle":edge_linestyle}
            _edge_kws.update(edge_kws)
            # Legend plotting
            _legend_kws = {}
            _legend_kws.update(legend_kws)

            # Edge info
            edges = self.kernel_subset_.data.copy()

            if not np.all(edges >= 0):
                warnings.warn("Adjacency weights must be ≥ 0 to plot.  Absolute values have been computed for the edgeweights before the `func_edgeweight` has been applied.")
                edges = edges.abs()
            if func_edgeweight is not None:
                edges = func_edgeweight(edges)
            if clip_edgeweight is not None:
                edges = np.clip(edges, a_min=None, a_max=clip_edgeweight)
                # edges[edges < 0] = np.clip(edges[edges < 0], a_min=-clip_edgeweight, a_max=None)
                # edges[edges > 0] = np.clip(edges[edges > 0], a_min=None, a_max=clip_edgeweight)
            if edge_colors is None:
                edge_colors = axis_color
            if is_color(edge_colors):
                edge_colors = dict_build([(edge_colors, edges.index)])
            if is_dict(edge_colors):
                edge_colors = pd.Series(edge_colors)
            if not is_color(edge_colors):
                if is_nonstring_iterable(edge_colors) and not isinstance(edge_colors, pd.Series):
                    edge_colors = pd.Series(edge_colors, index=edges.index)
            edge_colors = edge_colors[edges.index]

            # Plotting
            # =================
            for name_axis, axes_data in self.axes.items():
                # Retrieve
                node_positions = axes_data["node_positions_normalized"]
                colors = axes_data["colors"].tolist() # Needs `.tolist()` for Matplotlib version < 2.0.0
                sizes = axes_data["sizes"].tolist()


                # Positions
                # =========
                # Get a theta value for each node on the axis
                if not axes_data["split_axis"]:
                    theta_single = np.repeat(axes_data["theta"][0], repeats=node_positions.size)
                    theta_vectors = [theta_single]
                # Split the axis so within axis interactions can be visualized
                if axes_data["split_axis"]:
                    theta_split_A = np.repeat(axes_data["theta"][0], repeats=node_positions.size)
                    theta_split_B = np.repeat(axes_data["theta"][1], repeats=node_positions.size)
                    theta_vectors = [theta_split_A, theta_split_B]
                theta_representative = np.mean(axes_data["theta"])

                # Quadrant
                # =======
                if pad_axis_label == "infer":
                    pad_axis_label = 0.0618*(node_positions.max() - node_positions.min())
                quadrant, horizontalalignment, verticalalignment = self._get_quadrant_info(theta_representative)

                # Plot axis
                # =========
                if show_axis:
                    for theta in axes_data["theta"]:
                        ax.plot(
                            2*[theta],
                            [min(node_positions), max(node_positions)],
                            **_axis_kws,
                        )

                # Plot nodes
                # ========
                for theta in theta_vectors:
                    # Filled
                    ax.scatter(
                        theta,
                        node_positions,
                        c=axes_data["colors"],
                        s=axes_data["sizes"],
                        marker=axes_data["node_style"],
                        **_node_kws,
                    )
                    # Empty
                    ax.scatter(
                        theta,
                        node_positions,
                        facecolors='none',
                        s=axes_data["sizes"],
                        marker=axes_data["node_style"],
                        alpha=1,
                        zorder=_node_kws["zorder"]+1,
                        edgecolor=_node_kws["edgecolor"],
                        linewidth=_node_kws["linewidth"],
                    )

                # Plot axis labels
                # ================
                if show_axis_labels:
                    # Plot axis labels
                    ax.annotate(
                        s=name_axis,
                        xy=(theta_representative, node_positions.size + node_positions.max() + pad_axis_label),
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment,
                        **_axis_label_kws,
                    )
                # Plot node labels
                # ================
                if show_node_labels:
                    horizontalalignment_nodelabels = horizontalalignment
                    for name_node, r in node_positions.iteritems():
                        # theta_anchor is where the padding ends up
                        theta_anchor_padding = theta_representative
                        if node_pad_fill is None:
                            node_pad_fill = " "
                        node_pad_fill = str(node_pad_fill)
                        node_padding = node_pad_fill*pad_node_label
                        # Vertical axis case
                        condition_vertical_axis_left = (quadrant in {90,270}) and (node_label_position_vertical_axis == "left")
                        condition_vertical_axis_right = (quadrant in {90,270}) and (node_label_position_vertical_axis == "right")
                        if condition_vertical_axis_left:
                            horizontalalignment_nodelabels = "right" # These are opposite b/c nodes should be on the left which means padding on the right
                        if condition_vertical_axis_right:
                            horizontalalignment_nodelabels = "left" # Vice versa
                        # pad on the right and push label to left
                        if (quadrant == 3) or condition_vertical_axis_left:
                            name_node = f"{name_node}{node_padding}"
                            theta_anchor_padding = max(axes_data["theta"])
                        # pad on left and push label to the right
                        if (quadrant == 4) or condition_vertical_axis_right:
                            name_node = f"{node_padding}{name_node}"
                            theta_anchor_padding = min(axes_data["theta"])
                        # Label nodes
                        ax.annotate(
                            s=name_node,
                            xy=(theta_anchor_padding, r),
                            horizontalalignment=horizontalalignment_nodelabels,
                            verticalalignment="center",
                            **_node_label_kws,
                        )

            # Plot edges
            # ================
            # Draw edges
            if show_edges:
                if self.verbose:
                    print("-"*44, file=sys.stderr)
                    edges_iterable = tqdm(edges.iteritems(), "      Drawing edges")
                else:
                    edges_iterable = edges.iteritems()
                for (edge, weight) in edges_iterable:
                    node_A, node_B = edge
                    name_axis_A = self.node_mapping_[node_A]
                    name_axis_B = self.node_mapping_[node_B]

                    # Check axis
                    intraaxis_edge = (name_axis_A== name_axis_B)

                    # Within axis edges
                    if intraaxis_edge:
                        name_consensus_axis = name_axis_A
                        # Plot edges on split axis
                        if self.axes[name_consensus_axis]["split_axis"]:
                            # color = self.axes[name_consensus_axis]["colors"][node_A] #! Changed this 2019-06-03
                            color = edge_colors[edge]
                            # Draw edges between same axis
                            # Node A -> B
                            ax.plot([*self.axes[name_consensus_axis]["theta"]], # Unpack
                                    [self.axes[name_consensus_axis]["node_positions_normalized"][node_A], self.axes[name_consensus_axis]["node_positions_normalized"][node_B]],
                                    c=color,
                                    linewidth=weight,
                                    **_edge_kws,
                            )
                            # Node B -> A
                            ax.plot([*self.axes[name_consensus_axis]["theta"]], # Unpack
                                    [self.axes[name_consensus_axis]["node_positions_normalized"][node_B], self.axes[name_consensus_axis]["node_positions_normalized"][node_A]],
                                    c=color,
                                    linewidth=weight,
                                    **_edge_kws,
                           )

                    # Between axis
                    if not intraaxis_edge:
                        axes_ordered = list(self.axes.keys())
                        terminal_axis_edge = False
                        # Last connected to the first
                        if (name_axis_A == axes_ordered[-1]):
                            if (name_axis_B == axes_ordered[0]):
                                thetas = [self.axes[name_axis_A]["theta"].max(), self.axes[name_axis_B]["theta"].min()]
                                radii = [self.axes[name_axis_A]["node_positions_normalized"][node_A], self.axes[name_axis_B]["node_positions_normalized"][node_B]]
                                terminal_axis_edge = True
                        # First connected to the last
                        if (name_axis_A == axes_ordered[0]):
                            if (name_axis_B == axes_ordered[-1]):
                                thetas = [self.axes[name_axis_B]["theta"].max(), self.axes[name_axis_A]["theta"].min()]
                                radii = [self.axes[name_axis_B]["node_positions_normalized"][node_B], self.axes[name_axis_A]["node_positions_normalized"][node_A]]
                                terminal_axis_edge = True
                        if not terminal_axis_edge:
                            if axes_ordered.index(name_axis_A) < axes_ordered.index(name_axis_B):
                                thetas = [self.axes[name_axis_A]["theta"].max(), self.axes[name_axis_B]["theta"].min()]
                            if axes_ordered.index(name_axis_A) > axes_ordered.index(name_axis_B):
                                thetas = [self.axes[name_axis_A]["theta"].min(), self.axes[name_axis_B]["theta"].max()]
                            radii = [self.axes[name_axis_A]["node_positions_normalized"][node_A], self.axes[name_axis_B]["node_positions_normalized"][node_B]]

                        # Radii node positions
                        #
                        # Necessary to account for directionality of edge.
                        # If this doesn't happen then there is a long arc
                        # going counter clock wise instead of clockwise
                        # If straight lines were plotted then it would be thetas and radii before adjusting for the curve below
                        if terminal_axis_edge:
                            theta_end_rotation = thetas[0]
                            theta_next_rotation = thetas[1] + np.deg2rad(360)
                            thetas = [theta_end_rotation, theta_next_rotation]
                        # Create grid for thetas
                        t = np.linspace(start=thetas[0], stop=thetas[1], num=granularity)
                        # Get radii for thetas
                        radii = interp1d(thetas, radii)(t)
                        thetas = t

                        ax.plot(thetas,
                                radii,
                                c=edge_colors[edge],
                                linewidth=weight,
                                **_edge_kws,
                               )
            # Plot Legend
            # ===========
            if legend is not None:
                ax.legend(*format_mpl_legend_handles(legend), **_legend_kws)
            if title is not None:
                ax.set_title(title, **_title_kws, y=y)

            return fig, ax

    # Axis data
    def get_axis_data(self, name_axis=None, field=None):
        if name_axis is None:
            print(f"Available axes:", set(self.axes.keys()), file=sys.stderr)
        else:
            assert name_axis in self.axes, f"{name_axis} is not in the axes"

            df =  pd.DataFrame(dict_filter(self.axes[name_axis], ["colors", "sizes", "node_positions", "node_positions_normalized"]))
            if self.compiled:
                df["theta"] = [self.axes[name_axis]["theta"]]*df.shape[0]
            df.index.name = name_axis
            if field is not None:
                return df[field]
            else:
                return df
    # Connections
    def get_axis_connections(self, name_axis=None, include_self_loops=False, sort_by=None, ascending=False, return_multiindex=False):
        assert self.compiled == True, "Please `compile` before getting connections"
        if name_axis is not None:
            assert name_axis in self.axes, f"{name_axis} is not in the available axes for `name_axis`.  Please add and recompile or choose one of the available axes:\n{self.axes.keys()}"

        df_dense = self.kernel_subset_.to_dense(diagonal={True:None, False:0}[include_self_loops])
        df_connections = df_dense.groupby(self.node_mapping_, axis=1).sum()
        if name_axis is not None:
            idx_axis_nodes = self.axes[name_axis]["nodes"]
            df_connections = df_connections.loc[idx_axis_nodes,:]
            df_connections.index.name = name_axis
        if sort_by is not None:
            assert sort_by in self.axes, f"{sort_by} is not in the available axes for `sort_by`.  Please add and recompile or choose one of the available axes:\n{self.axes.keys()}"
            df_connections = df_connections.sort_values(by=sort_by, axis=0, ascending=ascending)
        if return_multiindex:
            df_connections.index = pd.MultiIndex.from_tuples(df_connections.index.map(lambda id_node: (self.node_mapping_[id_node], id_node)))
        return df_connections

    # Stats
    # =====
    def compare(self, kernel_query, func_stats=stats.mannwhitneyu, name_stat=None):
        """
        Compare the connections between 2 Hives or adjacencies using the specified axes assignments.
        """
        assert self.compiled == True, "Please `compile` before comparing adjacencies"

        if is_query_class(kernel_query, "Hive"):
            kernel_query = kernel_query.kernel
        if is_query_class(kernel_query, "Symmetric"):
            kernel_query = kernel_query.to_dense()
        assert is_symmetric(kernel_query, tol=1e-10)
        assert set(self.nodes_) <= set(kernel_query.index), f"kernel_query must contain all nodes from current Hive"
        df_adj_reference = self.to_dense()
        df_adj_query = kernel_query.loc[df_adj_reference.index, df_adj_reference.columns]
        d_statsdata = OrderedDict()
        if self.verbose:
            nodes_iterable = tqdm(df_adj_reference.index, "Computing stats for nodes and axes groupings")
        else:
            nodes_iterable = df_adj_reference.index

        # Get nodes
        d_statsdata = OrderedDict()
        for id_node in nodes_iterable:
            # Get axis groups
            stats_axes_data = list()
            for name_axis in self.axes:
                idx_axis_nodes = self.axes[name_axis]["nodes"]
                num_nodes = self.axes[name_axis]["num_nodes"]
                # Get comparison data
                u = df_adj_reference.loc[id_node,idx_axis_nodes]
                v = df_adj_query.loc[id_node,idx_axis_nodes]
                # Get stats
                stat, p = func_stats(u,v)
                if name_stat is None:
                    if hasattr(func_stats, "__name__"):
                        name_stat = func_stats.__name__
                    else:
                        name_stat = str(func_stats)
                # Store data
                row = pd.Series(OrderedDict([
                 ((name_axis, "num_nodes"), num_nodes),
                 ((name_axis, "∑(reference)"), u.sum()),
                 ((name_axis, "∑(query)"), v.sum()),
                 ((name_axis, name_stat), stat),
                 ((name_axis, "p_value"), p)
                ]))
                stats_axes_data.append(row)
            # Build pd.DataFrame
            d_statsdata[id_node] = pd.concat(stats_axes_data)
        return pd.DataFrame(d_statsdata).T

    # Exports
    # =======
    def as_graph(self):
        try:
            return self.graph
        except AttributeError:
            self.graph = self.kernel.as_graph()
            return self.graph

    def to_file(self, path:str, compression="infer"):
        write_object(self, path=path, compression=compression)
        return self

    def to_dense(self, node_subset=None):
        if node_subset is None:
            return self.kernel.to_dense()
        else:
            assert self.compiled == True, "Please `compile` before getting dense pd.DataFrame"
            df_dense = self.kernel_subset_.to_dense()
            if node_subset == True:
                return df_dense
            if node_subset in self.axes:
                nodes = self.axes[node_subset]["nodes"]
                return df_dense.loc[nodes,nodes]
            else:
                raise Exception("Unrecognized node_subset.  Please use None for entire adjacency, True for nodes used by axes, or an axis name.")

    def copy(self):
        return copy.deepcopy(self)


# =======================================================
# Connectivity
# =======================================================
# Connectivity
def connectivity(kernel, include_self_loops=False):
    """
    kernel can be either pd.DataFrame or Symmetric object
    clusters can be either pd.Series or dict object with keys as node and values as cluster e.g. {node_A:0}
    """
    # https://www.rdocumentation.org/packages/WGCNA/versions/1.63/topics/intramodularConnectivity
    # https://github.com/cran/WGCNA/blob/678c9da2b71c5b4a43cb55005224c243f411abc8/R/Functions.R
    if is_query_class(kernel, "Symmetric"):
        kernel = kernel.to_dense()

    kernel = kernel.copy()
    if not include_self_loops:
        np.fill_diagonal(kernel.values, 0)

    # Check symmetry
    assert is_symmetrical(kernel, tol=1e-8), "kernel must be symmetric.  Try using `soothsayer.utils.force_symmetry`"
    kernel = force_symmetry(kernel)

    #kTotal
    return kernel.sum(axis=1)
# Intramodular connectivty
def intramodular_connectivity(kernel, clusters:pd.Series, include_self_loops=False):
    """
    kernel can be either pd.DataFrame or Symmetric object
    clusters can be either pd.Series or dict object with keys as node and values as cluster e.g. {node_A:0}
    """
    # https://www.rdocumentation.org/packages/WGCNA/versions/1.63/topics/intramodularConnectivity
    # https://github.com/cran/WGCNA/blob/678c9da2b71c5b4a43cb55005224c243f411abc8/R/Functions.R
    if is_query_class(kernel, "Symmetric"):
        kernel = kernel.to_dense()
    if is_dict(clusters):
        clusters = pd.Series(clusters)
    kernel = kernel.copy()
    if not include_self_loops:
        np.fill_diagonal(kernel.values, 0)

    # Check clusters
    assert is_symmetrical(kernel, tol=1e-8), "kernel must be symmetric.  Try using `soothsayer.utils.force_symmetry`"
    kernel = force_symmetry(kernel)
    assert set(kernel.index) <= set(clusters.index), "All nodes (indices in kernel) must have a clustering assignment in `cluster`"
    clusters = clusters[kernel.index]
    data_connectivity = OrderedDict()

    #kTotal
    data_connectivity["kTotal"] = kernel.sum(axis=1)

    #kWithin
    _data = list()
    for cluster in clusters.unique():
        idx_attrs = clusters[lambda x: x == cluster].index
        kWithin__cluster = kernel.loc[idx_attrs,idx_attrs].sum(axis=1)
        _data.append(kWithin__cluster)
    data_connectivity["kWithin"] = pd.concat(_data)

    #kOut
    data_connectivity["kOut"] = data_connectivity["kTotal"] - data_connectivity["kWithin"]

    #kDiff
    data_connectivity["kDiff"] = data_connectivity["kWithin"] - data_connectivity["kOut"]

    return pd.DataFrame(data_connectivity)

# Unsigned to signed
def signed(X):
    """
    unsigned -> signed correlation
    """
    return (X + 1)/2

# Topological overlap
def topological_overlap_measure(adjacency, tol=1e-10):
    """
    Compute the topological overlap for a weighted adjacency matrix

    ====================================================
    Benchmark 5000 nodes (iris w/ 4996 noise variables):
    ====================================================
    TOM via rpy2 -> R -> WGCNA: 24 s ± 471 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    TOM via this function: 7.36 s ± 212 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    =================
    Acknowledgements:
    =================
    Original source:
        * Peter Langfelder and Steve Horvath
        https://www.rdocumentation.org/packages/WGCNA/versions/1.67/topics/TOMsimilarity
        https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-559

    Implementation adapted from the following sources:
        * Credits to @scleronomic
        https://stackoverflow.com/questions/56574729/how-to-compute-the-topological-overlap-measure-tom-for-a-weighted-adjacency-ma/56670900#56670900
        * Credits to @benmaier
        https://github.com/benmaier/GTOM/issues/3
    """
    # Compute topological overlap
    def _compute_tom(A):
        # Prepare adjacency
        np.fill_diagonal(A, 0)
        # Prepare TOM
        A_tom = np.zeros_like(A)
        # Compute TOM
        L = np.matmul(A,A)
        ki = A.sum(axis=1)
        kj = A.sum(axis=0)
        MINK = np.array([ np.minimum(ki_,kj) for ki_ in ki ])
        A_tom = (L+A) / (MINK + 1 - A)
        np.fill_diagonal(A_tom,1)
        return A_tom

    # Check input type
    node_labels = None
    if not isinstance(adjacency, np.ndarray):
        if is_query_class(adjacency, "Symmetric"):
            adjacency = adjacency.to_dense()
        assert np.all(adjacency.index == adjacency.columns), "`adjacency` index and columns must have identical ordering"
        node_labels = adjacency.index

    # Check input type
    assert is_symmetrical(adjacency, tol=tol), "`adjacency` is not symmetric"
    assert np.all(adjacency >= 0), "`adjacency` weights must ≥ 0"

    # Compute TOM
    A_tom = _compute_tom(np.asarray(adjacency))

    # Unlabeled adjacency
    if node_labels is None:
        return A_tom

    # Labeled adjacency
    else:
        return pd.DataFrame(A_tom, index=node_labels, columns=node_labels)

# Soft threshold curves
@check_packages(["WGCNA"], language="r", import_into_backend=False)
def determine_soft_threshold(similarity:pd.DataFrame, title=None, show_plot=True, query_powers = np.append(np.arange(1,10), np.arange(10,30,2)), style="seaborn-white", scalefree_threshold=0.85, pad=1.0, markeredgecolor="black"):
        """
        WGCNA: intramodularConnectivity
            function (similarity, RsquaredCut = 0.85, powerVector = c(seq(1,
                       10, by = 1), seq(12, 20, by <...> reNetworkConcepts, verbose = verbose,
                       indent = indent)
        returns fig, ax, df_sft
        """
        # Imports
        from ..r_wrappers.packages.WGCNA import pickSoftThreshold_fromSimilarity

        # Check
        if is_query_class(similarity, "Symmetric"):
            df_adj = similarity.to_dense()
        else:
            df_adj = similarity
        assert df_adj.index.value_counts().max() == 1, "Remove duplicate labels in row"
        assert df_adj.columns.value_counts().max() == 1, "Remove duplicate labels in columns"

        # Run pickSoftThreshold.fromSimilarity
        with Suppress(show_stdout=False, show_stderr=True):
            df_sft = pickSoftThreshold_fromSimilarity(df_adj, query_powers)
        df_sft["Scale-Free Topology Model Fit"] = -1*np.sign(df_sft.iloc[:,2])*df_sft.iloc[:,1]
        df_sft = df_sft.set_index("Power", drop=True)
        fig, ax = None, None
        if show_plot:
            with plt.style.context(style=style):
                fig, ax = plt.subplots(ncols=3, figsize=(20,5))
                df_plot = pd.DataFrame([pd.Series(df_sft.index, index=df_sft.index).astype(int), df_sft["Scale-Free Topology Model Fit"]], index=["x","y"]).T
                annot = list({(str(p), (p,r + 1e-2)) for p,r in df_sft["Scale-Free Topology Model Fit"].iteritems()})
                fig, ax[0], im = plot_scatter(ax=ax[0], data=df_plot, x="x", y = "y", c="teal", xlabel="power", edgecolor=markeredgecolor, linewidth=1, ylabel="Scale-Free Topology : $R^2$", title="Scale-Free Topology Model Fit", annot=annot, annot_kws={"fontsize":12})
                if scalefree_threshold:
                    ax[0].axhline(scalefree_threshold, color="red", linestyle=":", label=to_precision(scalefree_threshold, 3))
                    ax[0].legend(fontsize=12, title="Scale-Free Threshold")
                # Connectivity
                df_sft.loc[:,["mean.k.","median.k.", "max.k."]].plot(kind="line", marker="o", alpha=0.85, markeredgecolor=markeredgecolor, markeredgewidth=1, ax=ax[1])
                ax[1].set_title("Connectivity", fontsize=15, fontweight="bold")
                ax[1].set_ylabel("$k$", fontsize=15)
                ax[1].grid(False)

                # Density & Centralization & Heterogeneity
                np.log(df_sft.loc[:,["Density","Centralization", "Heterogeneity"]]).plot(kind="line", markeredgecolor=markeredgecolor, markeredgewidth=1, marker="o", alpha=0.85, ax=ax[2])
                ax[2].set_title("Network Stats", fontsize=15, fontweight="bold")
                ax[2].set_ylabel("log-scale", fontsize=15)
                ax[2].grid(False)

                if title:
                    fig.suptitle(title, fontsize=18, fontweight="bold", y=pad)
        return fig, ax, df_sft

# Cluster modularity matrix
def cluster_modularity(df:pd.DataFrame, node_type="node", iteration_type="iteration"):
    """

    n_louvain = 100

    louvain = dict()
    for rs in tqdm(range(n_louvain), "Louvain"):
        louvain[rs] = community.best_partition(graph_unsigned, random_state=rs)
    df = pd.DataFrame(louvain)

    # df.head()
    # 	0	1	2	3	4	5	6	7	8	9
    # a	0	0	0	0	0	0	0	0	0	0
    # b	1	1	1	1	1	1	1	1	1	1
    # c	2	2	2	2	2	2	2	2	2	2
    # d	3	3	3	3	3	3	3	3	3	3
    # e	4	1	1	4	1	4	4	1	4	1

    cluster_modularity(df).head()
    iteration  0  1  2  3  4  5  6  7  8  9
    node
    (b, a)     0  0  0  0  0  0  0  0  0  0
    (c, a)     0  0  0  0  0  0  0  0  0  0
    (d, a)     0  0  0  0  0  0  0  0  0  0
    (e, a)     0  0  0  0  0  0  0  0  0  0
    (a, f)     0  0  0  0  0  0  0  0  0  0
    """

    # Adapted from @code-different:
    # https://stackoverflow.com/questions/58566957/how-to-transform-a-dataframe-of-cluster-class-group-labels-into-a-pairwise-dataf


    # `x` is a table of (n=nodes, p=iterations)
    nodes = df.index
    iterations = df.columns
    x = df.values
    n,p = x.shape

    # `y` is an array of n tables, each having 1 row and p columns
    y = x[:, None]

    # Using numpy broadcasting, `z` contains the result of comparing each
    # table in `y` against `x`. So the shape of `z` is (n x n x p)
    z = x == y

    # Reshaping `z` by merging the first two dimensions
    data = z.reshape((z.shape[0] * z.shape[1], z.shape[2]))

    # Redundant pairs
    redundant_pairs = list(map(lambda node:frozenset([node]), nodes))

    # Create pairwise clustering matrix
    df_pairs = pd.DataFrame(
        data=data,
        index=pd.Index(list(map(frozenset, itertools.product(nodes,nodes))), name=node_type),
        columns=pd.Index(iterations, name=iteration_type),
        dtype=int,
    ).drop(redundant_pairs, axis=0)


    return df_pairs[~df_pairs.index.duplicated(keep="first")]

# Temporal Networks
class TemporalNetwork(object):
    """
    Experimental

    Future:
    * Should I require identifiers for `add_timepoint`? It would be much easier adding metadata.
    * Add categoricals for add_track.  Right now, they have to be barcharts.

    Usage:

    # Set pad for continuous track data
    whz_pad = 0.1
    whz_lim = (ds_perturbations.metadata_observations["whz"].min()-whz_pad, ds_perturbations.metadata_observations["whz"].max()+whz_pad )

    # Get timepoints for a particular subject
    for id_subject, perturbations in tqdm(ds_perturbations[None].loc[:,edges_union].groupby(ds_perturbations.metadata_observations["subject"])):
        # Perturbation profiles
        perturbations = perturbations.dropna(how="any", axis=0)
        # Sort the timepoints
        days = ds_abundance.metadata_observations.loc[perturbations.index, "day"].astype(int).sort_values()
        perturbations = perturbations.loc[days.index]
        # Only include subjects that have 3 timepoints (maximum for this dataset)
        if perturbations.shape[0] == 3:
            # Create temporal graph object
            temporal_graph = TemporalNetwork(id_subject, edge_type="perturbation",time_unit="day" )
            # Add each visit
            visits = list()
            for i, (id_visit, timepoint) in enumerate(days.iteritems(), start=0):
                connections = perturbations.loc[id_visit].dropna()
                connections = connections[lambda x: x != 0]
                temporal_graph.add_timepoint(t=timepoint, data=connections, id=id_visit) #! Should id be required?
                visits.append(id_visit)
            temporal_graph.compile()

            # Transform CLR -> MixedLM residuals to the real, fill missing values with zeros
            df_node_sizes = np.exp(mixedlm.residuals_.loc[visits]).T.fillna(0)
            # Relabel sample identifiers to timepoint labels #! The best way to do this?
            df_node_sizes.columns = df_node_sizes.columns.map(lambda id_visit: ds_perturbations.metadata_observations.loc[id_visit, "day"])

            # Get timepoint colors for the track
            track_colors = chr_status.obsv_colors[visits].copy()
            track_colors.index = track_colors.index.map(lambda id_visit: ds_perturbations.metadata_observations.loc[id_visit, "day"])
            # Add tracks (only one in this example)
            for name in ["whz"]:
                track_data = ds_perturbations.metadata_observations.loc[visits,name].copy()
                track_data.index = track_data.index.map(lambda id_visit: ds_perturbations.metadata_observations.loc[id_visit, "day"])
                temporal_graph.add_track(name=name.upper(),
                                         data=track_data,
                                         plot_type="bar",
                                         color=track_colors,
                                         ylim=whz_lim,
                )

            # Plot arcs
            fig, axes = temporal_graph.plot_arcs(node_sizes=df_node_sizes*5, show_timepoint_identifiers=True, show_tracks=True)
            fig.savefig("../Figures/TemporalNetworks/{}.pdf".format(id_subject), bbox_inches="tight")
    """
    # Initialize
    def __init__(self, name=None, description=None, time_unit=None, node_type=None, edge_type=None, experimental_warning=True, **metadata):
        if experimental_warning:
            warnings.warn("`TemporalGraph` is experimental.  Functionality must be evaluated, plotting functions, must be coded, and much more.  In the current state, this can be used to create create global temporal networks via`teneto.TemporalNetwork` and `nx.OrderedDiGraph` in addition to each timepoint-specific undirected network via `nx.Graph`.")
        self.name = name
        self.description = description
        self.time_unit = time_unit
        self.node_type = node_type
        self.edge_type = edge_type
        self.graphs = dict()
        self.metadata = metadata; self.metadata["num_tracks"] = 0
        self.intertemporal_connections_ = None
        self.intratemporal_connections_ = dict()
        self.timepoint_identifiers_ = dict()
        self.tracks = OrderedDict()
        self.compiled = False
    # Add timepoint
    def add_timepoint(self, t, data, id=None, **additional_edge_attributes):
        """
        """
        if self.compiled:
            print("You are adding new timepoints to a compiled TemporalGraph.  Please recompile.", file=sys.stderr)
            self.compiled = False
        assert is_number(t), "`t` must be a numeric type"
        if isinstance(data, pd.DataFrame):
            data = dense_to_condensed(data)

        self.intratemporal_connections_[t] = data.copy() #! Is this a good idea to copy and have pos/neg weights?
        graph = nx.Graph(name=t)
        for edge, connection in data.items():
            edge_attrs = {"weight":abs(connection), "sign":np.sign(connection)}
            for (field, data) in additional_edge_attributes.items():
                edge_attrs[field] = data[edge]
            graph.add_edge(*tuple(edge),**edge_attrs)

        self.timepoint_identifiers_[t] = id
        self.graphs[t] = graph

    # Add tracks
    def add_track(self, name, data, alpha=0.618, color="black", pad=0.1618, percent_of_figure = 20, ylim=None, plot_type="bar", spines=["top", "bottom", "left", "right"], fillna=0, plot_kws=dict(), label_kws=dict(),  bar_width="auto", check_missing_values=False, convert_to_hex=False, horizontal_kws=dict(), horizontal_lines=[]):
        """
        type(track) should be a pd.Series
        """
        assert self.compiled, "Please compile to continue."

        # Keywords
        if plot_type == "bar":
            _plot_kws = {"linewidth":1.382, "alpha":alpha, "width":max(0.1618, len(self.timepoints_)/30) if bar_width == "auto" else bar_width}
        elif plot_type == "area":
            _plot_kws = {"linewidth":0.618, "alpha":0.5}
        else:
            _plot_kws = {}
        _plot_kws.update(plot_kws)
        _horizontal_kws = {"linewidth":1, "linestyle":"-", "color":"black"}
        _horizontal_kws.update(horizontal_kws)
        _label_kws = {"fontsize":15, "rotation":0,  "ha":"right", "va":"center"}
        _label_kws.update(label_kws)

        # Color of the track
        if color is None:
            color = "black"
        if isinstance(color,str):
            color = pd.Series([color]*len(self.timepoints_), index=self.timepoints_)
        if is_dict(color):
            color = pd.Series(color, index=index)
        if convert_to_hex:
            _sampleitem_ = color[0]
            assert mpl.colors.is_color_like(_sampleitem_), f"{_sampleitem_} is not color-like"
            if is_rgb_like(_sampleitem_):
                color = color.map(rgb2hex)
        # Check for missing values
        if check_missing_values:
            assert set(data.index) == set(self.timepoints_), "`data.index` is not the same set as `self.timepoints_`"
            assert set(color.index) == set(self.timepoints_), "`color.index` is not the same set as `self.timepoints_`"
        if plot_type in ["area","line"]:
            if color.nunique() > 1:
                print("Warning: `area` and `line` plots can only be one color.  Defaults to first color in iterable.", file=sys.stderr)
        # Horizontal lines
        if is_number(horizontal_lines):
            horizontal_lines = [horizontal_lines]
        # Convert to ticks
        data = data[self.timepoints_]
        color = color[self.timepoints_]
        # Store data
        self.tracks[name] = {"data":data, "color":color, "pad":pad, "size":f"{percent_of_figure}%", "plot_type":plot_type, "plot_kws":_plot_kws,  "ylim":ylim, "horizontal_lines":horizontal_lines, "horizontal_kws":_horizontal_kws, "spines":spines, "label_kws":_label_kws}
        self.metadata["num_tracks"] += 1

        return self

    def __repr__(self):
        header = "TemporalGraph(name = {}):".format(self.name)

        return "{}\n\t* Timepoints: {} \
                \n\t* Compiled: {} \
                \n\t* Tracks: {} \
                ".format(
                    format_header(header),
                    sorted(self.graphs.keys()),
                    self.compiled,
                    self.metadata["num_tracks"],
                )

    # Compile
    @check_packages(["teneto"], language="python", import_into_backend=False)
    def compile(self, intertemporal_connections=None, check_intermodal=True, verbose=False, **additional_edge_attributes):
        from teneto import TemporalNetwork as TenetoTemporalNetwork

        # Get nodes and intratemporal edges
        self.node_temporal_ = defaultdict(set)
        for (t, graph) in self.graphs.items():
            for node in graph.nodes():
                self.node_temporal_[node].add(t)
        self.nodes_ = sorted(self.node_temporal_)
        self.number_of_nodes_ = len(self.nodes_)
        # No self.number_of_edges_ because we need to distinguish betweenn intratemporal and intertemporal

        self.intratemporal_edge_labels_ = pd.Index([*map(frozenset,itertools.combinations(self.nodes_, 2))], name=self.name)
        self.timepoints_ = np.asarray(sorted(self.graphs))


        # Encode nodes for teneto
        self.encoding_ = {node:i for i, node in enumerate((self.nodes_), start=0)}
        self.decoding_ = {i:node for node, i in self.encoding_.items()}

        # Temporal networks
        if verbose:
            print(format_header("Adding intratemporal edges:", "-"), file=sys.stderr)
        self.teneto_network_ = TenetoTemporalNetwork(desc=self.description, nettype="wu", starttime=min(self.graphs), N=len(self.nodes_), T=len(self.graphs))
        self.temporal_graph_ = nx.OrderedDiGraph(name=self.name)
        encoded_edges = list()
        decoded_edges = list()
        for t, graph in self.graphs.items():
            if verbose:
                iterable = tqdm(graph.edges(data=True), "t={}".format(t))
            else:
                iterable = graph.edges(data=True)
            for node_i,node_j, attrs in iterable:
                # Teneto
                edge_data = [self.encoding_[node_i], self.encoding_[node_j], t, attrs["weight"]]
                encoded_edges.append(edge_data)
                # NetworkX
                edge_data = [(node_i, t), (node_j, t), attrs]
                decoded_edges.append(edge_data)
                edge_data = [(node_j, t), (node_i, t), attrs]
                decoded_edges.append(edge_data)
        self.intratemporal_connections_  = pd.DataFrame(self.intratemporal_connections_ )
        self.intratemporal_connections_.index.name = "Edge"
        self.intratemporal_connections_.columns.name = "$t$"

        # Intertemporal connections
        self.intertemporal_connections_ = intertemporal_connections
        if intertemporal_connections is not None:
            # Verbose but quicker
            if check_intermodal:
                for edge, connection in tqdm(intertemporal_connections.items(), "Adding intertemporal edges"):
                    node_i_t_n, node_i_t_n1 = edge
                    t_n = node_i_t_n[-1]
                    t_n_1 = node_i_t_n1[-1]
                    assert t_n < t_n_1, "Please make sure t_n < t_n+1: {} | {} !< {}".format(edge, t_n, t_n_1)
                    assert node_i_t_n[0] == node_i_t_n1[0], "Please make sure `node_i_t_n` == `node_i_t_n1+1`: {} | {} !< {}".format(edge, node_i_t_n[0], node_i_t_n1[0])

                    edge_attrs = {"weight":abs(connection), "sign":np.sign(connection)}
                    for (field, data) in additional_edge_attributes.items():
                        edge_attrs[field] = data[edge]
                    edge_data = [node_i_t_n, node_i_t_n1, edge_attrs]
                    decoded_edges.append(edge_data)
            else:
                for edge, connection in tqdm(intertemporal_connections.items(), "Adding intertemporal edges"):
                    edge_attrs = {"weight":abs(connection), "sign":np.sign(connection)}
                    for (field, data) in additional_edge_attributes.items():
                        edge_attrs[field] = data[edge]
                    edge = tuple(edge)
                    edge_data = [node_i_t_n, node_i_t_n1, edge_attrs]
                    edge_data = [*edge,edge_attrs]
                    decoded_edges.append(edge_data)

            self.intertemporal_connections_ = True

        # Tento
        self.teneto_network_.add_edge(encoded_edges)

        # NetworkX
        self.temporal_graph_.add_edges_from(decoded_edges)

        # Compiled
        self.compiled = True
        return self

    def get_intratemporal_connections(self, t=None, mode="connection"):
        assert_acceptable_arguments(mode, {"connection", "weight", "sign"})
        if t is not None:
            connections = self.intratemporal_connections_[t]
        else:
            connections = pd.DataFrame(self.intratemporal_connections_)
        if mode == "connection":
            return connections
        if mode == "weight":
            return connections.abs()
        if mode == "sign":
            return np.sign(connections)

        return df_intratemporal_connections
    def get_intertemporal_connections(self):
        return self.intertemporal_connections_

    def compute_measure(self, measure:str, level:str="global", **measure_kws):
        """
        Reference: https://teneto.readthedocs.io/en/latest/teneto.networkmeasures.html
        level: {global, time, node, edge}
        measure: bursty_coeff, fluctuability, intercontacttimes, local_variation, reachability_latency, shortest_temporal_path, sid, temporal_betweenness_centrality, temporal_closeness_centrality,temporal_efficiency, temporal_participation_coeff, topological_overlap, volatility}

        #! Clean up this code...
        """
        assert self.compiled, "Please compile to continue."
        calc_incompatible_measures = {"intercontacttimes", "local_variation", "shortest_temporal_path","temporal_degree_centrality","temporal_participation_coeff"}
        if measure not in calc_incompatible_measures:
            measure_kws["calc"] = level
        data =  self.teneto_network_.calc_networkmeasure(measure, **measure_kws)

        output = None
        show_warning = False
        if measure  in calc_incompatible_measures:
            show_warning = False
        else:
            if is_number(data):
                output = data
            else:
                data = np.asarray(data)
                if len(data.shape) in {1,2}:
                    if len(data.shape) == 1:
                        N = data.size
                        inferred = False
                        # Time
                        if N == len(self.graphs)-1:
                            output = pd.Series(data, index=[*map(lambda t_n: (self.timepoints_[t_n], self.timepoints_[t_n+1]), range(self.timepoints_.size-1))], name=measure)
                            inferred = True
                        # Nodes
                        if N == len(self.nodes_):
                            output = pd.Series(data, index=self.nodes_, name=measure)
                            inferred = True

                    if len(data.shape) == 2:
                        output = pd.Series(squareform(data), index=self.intratemporal_edge_labels_, name=measure)
                        inferred = True

                    if not inferred:
                        show_warning = True
                else:
                    show_warning = True
        if show_warning:
            warnings.warn("Unable to determine labels.  Returning unlabeled data.")
            output = data

        return output

    def _plot_track_mpl(self, name, divider, xlim, xticks, ylim, show_track_ticks, show_track_labels, track_label_kws):

        ax = divider.append_axes("bottom", size=self.tracks[name]["size"], pad=self.tracks[name]["pad"])
        data = self.tracks[name]["data"]
        c = self.tracks[name]["color"]
        kws = self.tracks[name]["plot_kws"]
        # Set timepoint positions
        timepoint_positions = pd.Series(range_like(self.timepoints_), self.timepoints_)
        data.index = data.index.map(lambda x:timepoint_positions[x])
        c.index = c.index.map(lambda x:timepoint_positions[x])

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
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        if self.tracks[name]["ylim"] is not None:
            ax.set_ylim(ylim)
        if not show_track_ticks:
            ax.set_yticklabels([])
        if show_track_labels:
            ax.set_ylabel(name, **track_label_kws)
        return ax

    # Arc plots
    def plot_arcs(self,
                  # Network
                  nodelist=None,
                  node_positions=None,
                  node_colors=None,
                  node_outline_color=None,
                  edge_colors=None,
                  node_relabel=None,
                  node_sizes=100,
                  edge_linestyle="-",
                  edgecolor_negative='#278198',
                  edgecolor_positive='#dc3a23',

                  # Show
                  show_nodes=True,
                  show_node_labels=True,
                  show_timepoints=True,
                  show_timepoint_identifiers=True,
                  show_xgrid=False,
                  show_ygrid=True,
                  show_border=False,
                  show_tracks=True,
                  show_track_ticks=True,
                  show_track_labels=True,

                  # Plotting
                  style="seaborn-white",
                  figsize=(13,8),
                  background_color=None,
                  ax=None,
                  edge_alpha=0.5,
                  node_alpha=0.75,
                  node_fontsize=12,
                  timepoint_fontsize=15,
                  xlabel="$t$",
                  ylabel=None,
                  title=True,

                  # Keywords
                  fig_kws=dict(),
                  label_kws = dict(),
                  node_kws=dict(),
                  edge_kws=dict(),
                  node_tick_kws=dict(),
                  timepoint_tick_kws=dict(),
                  grid_kws=dict(),
                  timepoint_identifier_kws=dict(),
                  title_kws=dict(),

                  # Miscellaneous
                  clip_edgeweight=5,
                  granularity=100,
                  func_edgeweight=None,
                  curve_scaling_factor=1,
                  pad_timepoint_identifier="auto",
                  pad_timepoints="auto",
                 ):

        """
        `node_sizes` and `node_colors` can either be a single value, a pd.DataFrame with index as nodes and columns as timepoints, or a nested dictionary with {timepoint: {node: value}}
        """
        # This code has been adapted from https://teneto.readthedocs.io/en/latest/_modules/teneto/plot/slice_plot.html#slice_plot

        assert self.compiled, "Please compile to continue."
        if style in ["dark",  "black", "night",  "sith", "sun"]:
            style = "dark_background"
        if style in ["light", "white", "day", "jedi", "moon"] :
            style = "seaborn-white"
        if node_outline_color is None:
            if style == "dark_background":
                node_outline_color = "white"
            else:
                node_outline_color = "black"
        # ========
        # Defaults
        # ========
        # Figure
        _fig_kws = {"figsize":figsize}
        _fig_kws.update(fig_kws)

        # Title
        _title_kws = {"fontsize":18, "fontweight":"bold"}
        _title_kws.update(title_kws)

        # Axis label
        _label_kws = {"fontsize":15, "fontweight":"bold"}
        _label_kws.update(label_kws)

        # Timepoint ticks
        _timepoint_tick_kws = {"fontsize":timepoint_fontsize}
        _timepoint_tick_kws.update(timepoint_tick_kws)

        # Node ticks
        _node_tick_kws = {"fontsize":node_fontsize}
        _node_tick_kws.update(node_tick_kws)

        # Timepoint identifier
        _timepoint_identifier_kws = {"fontsize":12, "ha":"center", "va":"bottom"}
        _timepoint_identifier_kws.update(timepoint_identifier_kws)

        # Grids
        _grid_kws = {"linewidth":0.1618}
        _grid_kws.update(grid_kws)

        # Node plotting
        _node_kws = {"linewidth":1.618, "edgecolor":node_outline_color, "alpha":node_alpha, "zorder":1}
        _node_kws.update(node_kws)
        # Edge plotting
        _edge_kws = {"alpha":edge_alpha, "zorder":_node_kws["zorder"]+2, "linestyle":edge_linestyle}
        _edge_kws.update(edge_kws)

        # =====
        # Nodes
        # =====
        if nodelist is None:
            nodelist = self.nodes_
        num_query_nodes = len(nodelist)

        if node_positions is None:
            node_positions = pd.Series(range(num_query_nodes), nodelist)
        assert set(nodelist) <= set(self.nodes_), "Not all nodes in `nodelist` are in `self.nodes_`"
        node_positions = node_positions[nodelist]

        if node_relabel is None:
            node_relabel = dict(zip(nodelist, nodelist))

        # Node sizes
        # ----------
        # Default sizes
        if is_number(node_sizes):
            A = np.empty((num_query_nodes, len(self.timepoints_)))
            A[:] = node_sizes
            node_sizes = pd.DataFrame(A, index=nodelist, columns=self.timepoints_)
        node_sizes = pd.DataFrame(node_sizes)
        assert set(node_sizes.index) >= set(nodelist), "Not all nodes in `nodelist` have a size in `node_sizes`"
        node_sizes = node_sizes.loc[nodelist]

        # Node colors
         # ----------
        if node_colors is None:
            if style == "dark_background":
                node_colors = "white"
            else:
                node_colors = "black"
        # Default colors
        if is_color(node_colors):
            A = np.empty((num_query_nodes, len(self.timepoints_))).astype(object)
            A[:] = node_colors
            node_colors = pd.DataFrame(A, index=nodelist, columns=self.timepoints_)
        node_colors = pd.DataFrame(node_colors)
        assert set(node_colors.index) >= set(nodelist), "Not all nodes in `nodelist` have a size in `node_sizes`"
        node_colors = node_colors.loc[nodelist]

        # Edge colors
        # ----------
        if edge_colors is None:
            edge_colors = self.get_intratemporal_connections(mode="sign").astype(object)
            mask_positive = edge_colors.values == 1.0
            mask_negative = edge_colors.values == -1.0
            edge_colors[mask_positive] = edgecolor_positive
            edge_colors[mask_negative] = edgecolor_negative
            if style == "dark_background":
                edge_colors[edge_colors.values == 0] = "black"
            else:
                edge_colors[edge_colors.values == 0] = "white"


        # Plotting
        # ----------
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(**_fig_kws)
            else:
                fig = plt.gcf()
            # Axis collection
            axes = [ax]

            # Set ticks
            ax.set_xticks(range(len(self.timepoints_)))
            ax.set_yticks(node_positions.values)

            # Plot each timepoint
            for t_i, t in enumerate(self.timepoints_):

                # Plot nodes
                if show_nodes:
                    ax.scatter(
                            x=np.ones(len(nodelist))*t_i,
                            y=node_positions.values,
                            c=node_colors[t],
                            s=node_sizes[t],
                            **_node_kws,
                    )

                # Plot arcs
                edges = self.intratemporal_connections_[t].dropna().abs()
                edges = edges[lambda x: x > 0]

                if func_edgeweight is not None: # Tranform edge weights
                    edges = func_edgeweight(edges)
                if clip_edgeweight is not None:
                    edges = np.clip(edges, a_min=None, a_max=clip_edgeweight)
                edge_colors_for_timepoint = edge_colors[t] # Change name of this
                for edge, weight in edges.items():
                    node_a, node_b = list(edge)
                    bvx, bvy = bezier_points(
                        p1=(t_i, node_positions[node_a]),
                        p2=(t_i, node_positions[node_b]),
                        control=num_query_nodes*(1/curve_scaling_factor),
                        granularity=granularity,
                    )
                    ax.plot(bvx, bvy, linewidth=weight, color=edge_colors_for_timepoint[edge], **_edge_kws)


            tracks_visible = False
            if show_tracks:
                divider = make_axes_locatable(ax)
                for name in self.tracks:
                    if "edgecolor" in self.tracks[name]["plot_kws"]:
                        if self.tracks[name]["plot_kws"]["edgecolor"] == "auto":
                            self.tracks[name]["plot_kws"]["edgecolor"] = {True:"white", False:"black"}[style == "dark_background"]
                    args_track = dict(
                        name=name,
                        divider=divider,
                        xlim=ax.get_xlim(),
                        xticks=ax.get_xticks(),
                        ylim=self.tracks[name]["ylim"],
                        show_track_ticks=show_track_ticks,
                        show_track_labels=show_track_labels,
                        track_label_kws=self.tracks[name]["label_kws"],
                    )
                    ax_track = self._plot_track_mpl( **args_track)
                    ax.set_xticklabels([])

                    if show_xgrid:
                        ax_track.xaxis.grid(show_xgrid, **_grid_kws)
                    if show_ygrid:
                        ax_track.yaxis.grid(show_ygrid, **_grid_kws)
                    axes.append(ax_track)
                    tracks_visible = True


            # Show temporal identifiers
            if show_timepoint_identifiers:
                ax.autoscale(False)
                if all([tracks_visible, pad_timepoint_identifier == "auto"]):
                    pad_timepoint_identifier = -1
                else:
                    pad_timepoint_identifier = -0.382
                for t_i, t in enumerate(self.timepoints_):
                    id = self.timepoint_identifiers_[t]
                    if id is not None:
                        axes[-1].text(x=t_i, y=min(axes[-1].get_ylim()) + pad_timepoint_identifier, s=id, **_timepoint_identifier_kws)


            # Set tick labels
            if show_timepoints:
                if all([tracks_visible, pad_timepoints == "auto"]):
                    pad_timepoints = -0.5
                else:
                    pad_timepoints = -0.05

                axes[-1].set_xticklabels(self.timepoints_, y=pad_timepoints, **_timepoint_tick_kws)

            if show_node_labels:
                ax.set_yticklabels(list(map(lambda node: node_relabel[node] if node in node_relabel else node, nodelist)), **_node_tick_kws)

            # Grids and border
            if show_xgrid:
                ax.xaxis.grid(show_xgrid, **_grid_kws)
            if show_ygrid:
                ax.yaxis.grid(show_ygrid, **_grid_kws)
            if not show_border: # Not using ax.axis('off') becuase it removes facecolor
                for spine in ax.spines.values():
                    spine.set_visible(False)

            # Background color
            if background_color is not None:
                ax.set_facecolor(background_color)
            # Labels
            if xlabel:
                axes[-1].set_xlabel(xlabel, **_label_kws)
            if ylabel:
                ax.set_ylabel(ylabel, **_label_kws)
            # Title
            if title == True:
                title = self.name
            if title:
                ax.set_title(title, **_title_kws)

            return fig, axes

    def plot_network(self):
        assert self.compiled, "Please compile to continue."
        print("Not available in this version.", file=sys.stderr)
        pass

    def plot_hive(self):
        assert self.compiled, "Please compile to continue."
        print("Not available in this version.", file=sys.stderr)
        pass
