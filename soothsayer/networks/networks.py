
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
from scipy.interpolate import interp1d
from scipy import stats

# Soothsayer
from ..symmetry import *
from ..utils import *
from ..r_wrappers.packages.WGCNA import TOMsimilarity, pickSoftThreshold_fromSimilarity
from ..visuals import plot_scatter
from ..transmute.normalization import normalize_minmax
from ..io import write_object


__all__ = ["Hive", "intramodular_connectivity", "topological_overlap_measure", "signed", "determine_soft_threshold"]
__all__ = sorted(__all__)
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
        idx_attrs = clusters.compress(lambda x: x == cluster).index
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
def topological_overlap_measure(adjacency):
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
    assert is_symmetrical(adjacency, tol=1e-10), "`adjacency` is not symmetric"
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
def determine_soft_threshold(similarity:pd.DataFrame, title=None, show_plot=True, query_powers = np.append(np.arange(1,10), np.arange(10,30,2)), style="seaborn-white", scalefree_threshold=0.85, pad=1.0, markeredgecolor="black"):
        """
        WGCNA: intramodularConnectivity
            function (similarity, RsquaredCut = 0.85, powerVector = c(seq(1,
                       10, by = 1), seq(12, 20, by <...> reNetworkConcepts, verbose = verbose,
                       indent = indent)
        returns fig, ax, df_sft
        """
        # Check
        if is_query_class(similarity, "Symmetric"):
            df_adj = similarity.to_dense()
        else:
            df_adj = similarity
        assert df_adj.index.value_counts().max() == 1, "Remove duplicate labels in row"
        assert df_adj.columns.value_counts().max() == 1, "Remove duplicate labels in columns"

        # Run pickSoftThreshold.fromSimilarity
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
