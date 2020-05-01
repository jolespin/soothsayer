import os,sys,operator, warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.transforms as mtrans
import matplotlib_venn
import seaborn as sns
from adjustText import adjust_text
from collections import OrderedDict, defaultdict
from ..microbiome import prevalence
from ..utils import *

__all__ = ["OutlineCollection", "draw_networkx_labels_with_box", "plot_scatter", "plot_venn_diagram", "plot_waterfall", "plot_volcano", "plot_annotation", "plot_prevalence", "plot_compositional", "bezier_points"]
__all__ = sorted(__all__)

# Bezier points for arc plots
def bezier_points(p1, p2, control, granularity=20):
    """
    Credit: William Hedley Thompson
    https://teneto.readthedocs.io/en/latest/_modules/teneto/plot/slice_plot.html#slice_plot

    This has been adapted from the teneto package.

    # Following 3 Function that draw vertical curved lines from around points.
    # p1 nad p2 are start and end trupes (x,y coords) and pointN is the resolution of the points
    # negxLim tries to restrain how far back along the x axis the bend can go.
    """
    def pascal_row(n):
        # This returns the nth row of Pascal's Triangle
        result = [1]
        x, numerator = 1, n
        for denominator in range(1, n // 2 + 1):
            x *= numerator
            x /= denominator
            result.append(x)
            numerator -= 1
        if n & 1 == 0:
            # n is even
            result.extend(reversed(result[:-1]))
        else:
            result.extend(reversed(result))
        return np.asarray(result)
    # These two functions originated from the plot.ly's documentation for python API.
    # They create points along a curve.
    def make_bezier(points):
        # xys should be a sequence of 2-tuples (Bezier control points)
        n = len(points)
        combinations = pascal_row(n - 1)

        def bezier(ts):
            # This uses the generalized formula for bezier curves
            # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
            result = []
            for t in ts:
                tpowers = np.power(np.ones_like(n)*t, np.arange(n))
                upowers = np.power(np.ones_like(n)*(1-t), np.arange(n))[::-1]
                coefs = combinations*tpowers*upowers

                result.append(tuple(np.sum(coefs*ps) for ps in zip(*points))) # Need to optimize this part
            return np.asarray(result)
        return bezier

    ts = np.arange(granularity+1)/granularity
    d = p1[0] - (max(p1[1], p2[1]) - min(p1[1], p2[1])) / control
    points = np.asarray([p1, (d, p1[1]), (d, p2[1]), p2])
    bezier = make_bezier(points)
    points = bezier(ts)
    bvx = points[:,0]
    bvy = points[:,1]
    return np.asarray([bvx, bvy])

# Plot outline around line objects
class OutlineCollection(PatchCollection):
    """
    Source: @importanceofbeingernest
    https://stackoverflow.com/questions/55911075/how-to-plot-the-outline-of-the-outer-edges-on-a-matplotlib-line-in-python/56030879#56030879
    """
    def __init__(self, linecollection, ax=None, **kwargs):
        self.ax = ax or plt.gca()
        self.lc = linecollection
        assert np.all(np.array(self.lc.get_segments()).shape[1:] == np.array((2,2)))
        rect = plt.Rectangle((-.5, -.5), width=1, height=1)
        super().__init__((rect,), **kwargs)
        self.set_transform(mtrans.IdentityTransform())
        self.set_offsets(np.zeros((len(self.lc.get_segments()),2)))
        self.ax.add_collection(self)

    def draw(self, renderer):
        segs = self.lc.get_segments()
        n = len(segs)
        factor = 72/self.ax.figure.dpi
        lws = self.lc.get_linewidth()
        if len(lws) <= 1:
            lws = lws*np.ones(n)
        transforms = []
        for i, (lw, seg) in enumerate(zip(lws, segs)):
            X = self.lc.get_transform().transform(seg)
            mean = X.mean(axis=0)
            angle = np.arctan2(*np.squeeze(np.diff(X, axis=0))[::-1])
            length = np.sqrt(np.sum(np.diff(X, axis=0)**2))
            trans = mtrans.Affine2D().scale(length,lw/factor).rotate(angle).translate(*mean)
            transforms.append(trans.get_matrix())
        self._transforms = transforms
        super().draw(renderer)

# =====================
# NetworkX Wrappers
# =====================
def draw_networkx_labels_with_box(G, pos,
                         labels=None,
                         font_size=15,
                         font_color=None,
                         font_family='sans-serif',
                         font_weight='normal',
                         alpha=1.0,
                         node_box_facecolor=None,
                         node_box_edgecolor=None,
                         node_alpha=0.85,
                         boxstyle='round,pad=1',
                         linewidth=1.618,
                         ax=None,
                         **kwds):
    # https://github.com/networkx/networkx/blob/master/networkx/drawing/nx_pylab.py
    """
    __future___
    * Clean up the code for `d_nodeface_alpha`
    """
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = list(G.nodes())

    if not is_dict_like(labels):
        labels = dict((n, n) for n in labels)

    # Set colors
    def _process_colors(G, DEFAULT, colors):
        if colors is None:
            colors = DEFAULT
        if is_color_like(colors):
            colors = dict(zip(G.nodes(), G.number_of_nodes()*[colors]))
        if isinstance(colors, pd.Series):
            colors = colors.to_dict()
        return {n:colors[n] if n in colors else DEFAULT for n in G.nodes()}

    # Default
    DEFAULT_font_color = "#000000"
    DEFAULT_node_box_facecolor = "#d8d8d8"
    DEFAULT_node_box_edgecolor =  "#000000"
    DEFAULT_node_box_alpha = 0.85


    # Colors
    d_nodefont_color = _process_colors(G, DEFAULT_font_color, font_color)
    d_nodeface_color = _process_colors(G, DEFAULT_node_box_facecolor, node_box_facecolor)
    d_nodeedge_color = _process_colors(G, DEFAULT_node_box_edgecolor, node_box_edgecolor)
    # Alpha
    if is_number(node_alpha):
        d_nodeface_alpha = {node:node_alpha for node in d_nodeface_color}
    else:
        assert is_dict_like(node_alpha), "If `node_alpha` is not a single alpha value then it must be dict-like of {node_label:alpha}"
        d_nodeface_alpha = dict()
        for n, label in labels.items():
            if label in node_alpha:
                d_nodeface_alpha[n] = node_alpha[n]
            else:
                d_nodeface_alpha[n] = DEFAULT_node_box_alpha


    # set optional alignment
    horizontalalignment = kwds.get('horizontalalignment', 'center')
    verticalalignment = kwds.get('verticalalignment', 'center')

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        if n in pos:
            (x, y) = pos[n]
            if not nx.utils.is_string_like(label):
                label = str(label)  # this makes "1" and 1 labeled the same
            t = ax.text(x, y,
                        label,
                        size=font_size,
                        color=d_nodefont_color[n],
                        family=font_family,
                        weight=font_weight,
                        alpha=alpha,
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment,
                        transform=ax.transData,
                        bbox=dict(facecolor=d_nodeface_color[n], edgecolor=d_nodeedge_color[n], boxstyle=boxstyle, alpha=d_nodeface_alpha[n], linewidth=linewidth),
                        clip_on=True,
                        )
            text_items[n] = t
    return text_items
# ===================
# Matplotlib/Seaborn/Matplotlib_venn Wrappers
# ===================
# Scatter plots
def plot_scatter(
                 # Main
                 data:pd.DataFrame,
                 x=None,
                 y=None,
                 z=None,
                 c="gray",
                 s=None,
                 # Labels
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 zlabel=None,
                 cbar_label=None,
                 show_labels=None,
                 color_labels=None,
                 show_grid=False,
                 show_axis=True,

                 # Plotting params
                 figsize=(8,8),
                 ax=None,
                 edgecolor="white",
                 linewidth=1,
                 alpha=0.9,
                 logx=False,
                 logy=False,
                 logz=False,
                 legend=None,
                 annot=None,
                 annot_jitter=None,
                 cbar=True,
                 box_alpha=0.85,
                 auto_scale=True,

                 # Aspects
                 share_axis=False,
                 grid_granularity=None,
                 focus=None,
                 pad_focus="auto",
                 xlim=None,
                 ylim=None,
                 zlim=None,
                 cmap = None,
                 vmin=None,
                 vmax=None,
                 style="seaborn-white",
                 dimensions=2,

                 # Ellipse
                 ellipse_data=None,
                 ellipse_linewidth=3,
                 ellipse_linestyle="-",
                 ellipse_fillcolor="gray",
                 ellipse_edgecolor="black",
                 ellipse_n_std = 3,
                 ellipse_alpha=0.333,
                 ellipse_fill=False,

                 # 3D-Specific Parameters
                 elev_3d=None,
                 azim_3d=None,
                 auto_scale_3d=True,
                 pane_color_3d = None,

                 # Keywords
                 fig_kws=dict(),
                 annot_kws=dict(),
                 legend_kws=dict(),
                 cbar_kws=dict(),
                 axis_kws=dict(fontsize=15),
                 title_kws=dict(),
                 label_kws=dict(),

                 # Misc
                 pseudocount=1e-5,
                 missing_color='#FFFFF0',
                 missing_alpha=1.0,
                 verbose=True,
                 func_centroid=np.mean,
                 **args
                 ):
    """
    annot: [(text, (x,y))]
    Future:
        Fix the annotations so it's (x,y): annot_text and incorporate data.index (i.e. show_labels?)
        Add a polar coordinate conversion option
    09.26.2017
        Updated to clean up color mapping.  Force all alphas as None and if alpha is supplied then added within rgba
    10.02.2017
        Adding 3D support. Warning `lims` may not work.  Removing option for not having `data`.  Created `configure_scatter_utility`
        to do preprocessing.
    """
    # Keywords
    if figsize=="slides":
        figsize=(26.18,  14.7)
    _fig_kws = dict(figsize=figsize)
    _fig_kws.update(fig_kws)
    _legend_kws = {'bbox_to_anchor': (1, 0.5), 'edgecolor': 'black', 'facecolor': 'white', 'fontsize': 15, 'frameon': True, 'loc': 'center left'}
    _legend_kws.update(legend_kws)
    _cbar_kws = dict()
    _cbar_kws.update(cbar_kws)
    _title_kws = dict(fontsize=15,fontweight="bold")
    _title_kws.update(title_kws)
    _annot_kws = {**axis_kws,  "color":"black", "bbox":{"facecolor":"white", "edgecolor":"black", "alpha":box_alpha, "linewidth":2}, }
    _annot_kws.update(annot_kws)

    # ==================================================
    # Configure plot input data
    # ==================================================
    args_utility = dict(
                         data=data,
                         x=x,
                         y=y,
                         z=z,
                         c=c,
                         s=s,
                         alpha=alpha,
                         # Aspects
                         cmap = cmap,
                         vmin=vmin,
                         vmax=vmax,
                         dimensions=dimensions,
                         # Missing data
                         missing_color=missing_color,
                         missing_alpha=missing_alpha,
                         verbose=verbose,)

    configured_plot_data = configure_scatter(**args_utility)
    # Unpack configured data
    coords = configured_plot_data["coords"]
    c = np.asarray(configured_plot_data["colors"])
    s = np.asarray(configured_plot_data["sizes"])
    cmap = configured_plot_data["cmap"]
    scalar_vector = np.asarray(configured_plot_data["scalar_vector"])

    # Unpack Coordinates
    x = np.asarray(coords["x"])
    y = np.asarray(coords["y"])
    if dimensions == 3:
        z = np.asarray(coords["z"])

    # ==================================================
    # Log scale
    # ==================================================
    if logx:
        x = np.log(x + pseudocount)
        if xlabel:
            xlabel = "log(%s)"%xlabel
    if logy:
        y = np.log(y + pseudocount)
        if ylabel:
            ylabel = "log(%s)"%ylabel
    if logz:
        z = np.log(z + pseudocount)
        if zlabel:
            zlabel = "log(%s)"%zlabel

    with plt.style.context(style):
        # Matplotlib
        if ax is None:
            fig = plt.figure(**_fig_kws)
            if dimensions == 2:
                ax = plt.subplot(111, polar=False)
            if dimensions == 3:
                ax = fig.add_subplot(111, projection="3d")
        else:
            fig = plt.gcf()

        # ==================================================
        # Polar transformation
        # ==================================================
#         if (polar == True) and (polar_transform_x):
#             x = linear2circular(x)
        # ==================================================
        # Plot
        # ==================================================
        if dimensions == 2:
            im = ax.scatter(x=x, y=y, c=c, s=s, edgecolor=edgecolor, linewidth=linewidth, alpha=None, cmap=cmap, **args)

        if dimensions == 3:
            im = ax.scatter(xs=x, ys=y, zs=z, c=c, s=s, edgecolor=edgecolor, linewidth=linewidth, alpha=None, cmap=cmap, **args)

            # Change angle
            ax.view_init(elev=elev_3d, azim=azim_3d)
            # Change pane color
            if pane_color_3d:
                ax.w_xaxis.set_pane_color(pane_color_3d)
                ax.w_yaxis.set_pane_color(pane_color_3d)
                ax.w_zaxis.set_pane_color(pane_color_3d)

            # Auto scaling
            if auto_scale_3d:
                # X
                scaled_xlim_min = x.min() - x.min()*1e-3
                scaled_xlim_max = x.max() + x.min()*1e-3
                scaled_xlims = (scaled_xlim_min, scaled_xlim_max)
                # Y
                scaled_ylim_min = y.min() - y.min()*1e-3
                scaled_ylim_max = y.max() + y.min()*1e-3
                scaled_ylims = (scaled_ylim_min, scaled_ylim_max)

                # Z
                scaled_zlim_min = z.min() - z.min()*1e-3
                scaled_zlim_max = z.max() + z.min()*1e-3
                scaled_zlims = (scaled_zlim_min, scaled_zlim_max)

                # Set
                ax.set_xlim(scaled_xlims)
                ax.set_ylim(scaled_ylims)
                ax.set_zlim(scaled_zlims)

        # ==================================================
        # Labels
        # ==================================================
        if title:
            ax.set_title(str(title), **_title_kws)
        if xlabel:
            ax.set_xlabel(str(xlabel), **axis_kws)
        if ylabel:
            ax.set_ylabel(str(ylabel), **axis_kws)
        if zlabel:
            ax.set_zlabel(str(zlabel), **axis_kws)
        # ==================================================
        # Grid
        # ==================================================
        ax.grid(show_grid)
        if grid_granularity is not None:
            ax.set_xticks(np.linspace(*ax.get_xlim(), grid_granularity))
            ax.set_yticks(np.linspace(*ax.get_ylim(), grid_granularity))
        if show_axis == False:
            ax.set_axis_off()
        # ==================================================
        # Annotations
        # ==================================================
        if annot is not None:
            # Make this handle other colors
            if isinstance(annot, pd.Series):
                annot = annot.to_dict()
            if is_dict(annot):
                annot = list(annot.items())

            # Format the data into a list
            # NOTE! Currently cannot have duplicate indicies or {s:xy1, s:xy2}
            # ___________________________
            # Is data a list of strings to index?
            if not is_nonstring_iterable(annot[0]):
                annot = [(k, xy.as_matrix()) for k, xy in coords.loc[annot,:].T.iteritems()]
            # Is data a {s:[id_1, id_2, ..., id_m]} object?
                # Assumes the lists are greater than 2
            if is_nonstring_iterable(annot[0][1]) and (len(annot[0][1]) > 2):
                annot = get_coords_centroid(coords, annot, metric=func_centroid)

            for i,(annot_text, (x_coord,y_coord)) in enumerate(annot):
                xy = np.array((x_coord,y_coord))
                if annot_jitter:
                    xy += np.random.RandomState(i).normal(loc=0, scale=annot_jitter, size=1)
                ax.annotate(annot_text, xy, **_annot_kws)
        # ==================================================
        # Ellipses
        # ==================================================
        if ellipse_data is not None:
            # Format Data
            if not isinstance(ellipse_data, pd.DataFrame):
                ellipse_data = get_parameters_ellipse(coords, groups=ellipse_data, metric=func_centroid, n_std=ellipse_n_std)

            # Default Ellipse Attributes
            # Edge Color
            if (not isinstance(ellipse_edgecolor, pd.Series)) and (not is_dict(ellipse_edgecolor)):
                if not is_color(ellipse_edgecolor):
                    ellipse_edgecolor = "black"
                    if verbose:
                        print(f"{ellipse_edgecolor} is invalid.  Defaulting to `black`", file=sys.stderr)
            if is_color(ellipse_edgecolor):
                ellipse_edgecolor = pd.Series([ellipse_edgecolor]*ellipse_data.shape[0], index=ellipse_data.index, name="ellipse_edgecolor")
            if is_dict(ellipse_edgecolor):
                ellipse_edgecolor = pd.Series(ellipse_edgecolor, name="ellipse_edgecolor")
            # Fill Color
            if (not isinstance(ellipse_fillcolor, pd.Series)) and (not is_dict(ellipse_fillcolor)):
                if not is_color(ellipse_fillcolor):
                    ellipse_fillcolor = "black"
                    if verbose:
                        print(f"{ellipse_fillcolor} is invalid.  Defaulting to `black`", file=sys.stderr)
            if is_color(ellipse_fillcolor):
                ellipse_fillcolor = pd.Series([ellipse_fillcolor]*ellipse_data.shape[0], index=ellipse_data.index, name="ellipse_fillcolor")
            if is_dict(ellipse_fillcolor):
                ellipse_fillcolor = pd.Series(ellipse_fillcolor, name="ellipse_fillcolor")


            # Linestyle
            if type(ellipse_linestyle) == str:
                ellipse_linestyle = pd.Series([ellipse_linestyle]*ellipse_data.shape[0], index=ellipse_edgecolor.index, name="ellipse_linestyle")
            # Linewidth
            if is_number(ellipse_linewidth):
                ellipse_linewidth = pd.Series([ellipse_linewidth]*ellipse_data.shape[0], index=ellipse_edgecolor.index, name="ellipse_linewidth")
            for group, Se_data in ellipse_data.iterrows():
                ellipse_object = Ellipse(xy=Se_data[["x","y"]],
                                         width=Se_data["width"],
                                         height=Se_data["height"],
                                         angle=Se_data["theta"],
                                         fill=ellipse_fill,
                                         edgecolor=ellipse_edgecolor[group],
                                         facecolor=ellipse_fillcolor[group],

                                         linewidth=ellipse_linewidth[group],
                                         linestyle=ellipse_linestyle[group],
                                         alpha=ellipse_alpha,
                                            )
                ax.add_artist(ellipse_object)
        # ==================================================
        # Point labels
        # ==================================================
        if show_labels:
            # Coloring labels
            if is_dict(color_labels):
                color_labels = pd.Series(color_labels)
            # Default to black
            if color_labels is None:
                color_labels = "black"
            # Color all values same color
            if type(color_labels) == str:
                color_labels = pd.Series([color_labels]*len(x), index=data.index)
            # If True, set colors to match scatter points
            if not isinstance(color_labels, pd.Series):
                if color_labels == True:
                    color_labels = c.copy()
            # Default all the labels if True
            if type(show_labels) == bool:
                if show_labels == True:
                    show_labels = data.index
            # Plot Text
            for obsv_id in show_labels:
                if obsv_id in data.index:
                    if dimensions == 2:
                        ax.text(x=coords.loc[obsv_id, "x"],
                                y=coords.loc[obsv_id, "y"],
                                s=obsv_id,
                                color=color_labels[obsv_id],
                                **label_kws)
                    if dimensions == 3:
                        ax.text(x=coords.loc[obsv_id, "x"],
                                y=coords.loc[obsv_id, "y"],
                                z=coords.loc[obsv_id, "z"],
                            s=obsv_id,
                            color=color_labels[obsv_id],
                            **label_kws)
        # ==================================================
        # Sharing axis
        # ==================================================
        # Match axis scales
        if share_axis == True:
            lims = coords.as_matrix().ravel()
            min_lim, max_lim = min(lims), max(lims)
            ax.set_xlim((min_lim*0.98,max_lim*1.02))
            ax.set_ylim((min_lim*0.98,max_lim*1.02))
            if dimensions == 3:
                ax.set_zlim((min_lim*0.98,max_lim*1.02))

        # ==================================================
        # Legend
        # ==================================================
        if legend is not None:
            ax.legend(*format_mpl_legend_handles(legend), **_legend_kws)
        # ==================================================
        # Focus on an area
        # ==================================================
        if focus:
            # Not tested for 3D
            if type(focus) == str:
                focus = data.loc[focus,:]
            if pad_focus == "auto":
                sub = lambda x: x[0] - x[1]
                pad_focus = (sub(ax.get_xlim()) + sub(ax.get_ylim()))*0.01

            ax.set_xlim((focus[0] - pad_focus, focus[0] + pad_focus))
            ax.set_ylim((focus[1] - pad_focus, focus[1] + pad_focus))
        # ==================================================
        # Limits
        # ==================================================
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if dimensions == 3:
            if zlim:
                ax.set_zlim(zlim)
        # ==================================================
        # Autoscaling
        # ==================================================
        if auto_scale:
            ax.autoscale_view()
        # ==================================================
        # Colorbar
        # ==================================================
        if cbar:
            if cmap is not None:
                try:
                    add_cbar_from_data(fig=fig, cmap=cmap, data=scalar_vector, label=cbar_label, cbar_kws=cbar_kws, vmin=vmin, vmax=vmax)
                except AssertionError:
                    print("Warning: Colorbar was not added because `vmin` and `vmax` are Nonetype and are necessary when color values are not scalar", file=sys.stderr)
        return fig, ax, im

# Venn diagrams
def plot_venn_diagram(subsets, labels=None, colors=None, palette="Set1", edgecolor="white", linestyle="-", linewidth=1.618, fontsize_labels = 15, fontsize_numbers=15, alpha=0.8, style="seaborn-white", normalize_to=1.0, desat=None, figsize=(5,5), ax=None, circ_kws=dict(), title=None, title_kws=dict()):
    """
    return fig, ax, (venn, circ)
    """
    with plt.style.context(style):

        # Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        # Keywords
        _circ_kws = dict(alpha=1.0, color=edgecolor, linestyle=linestyle, linewidth=linewidth, ax=ax)
        _circ_kws.update(circ_kws)
        _title_kws = dict(fontsize=15, fontweight="bold")
        _title_kws.update(title_kws)

        # Force set type
        subsets = [*map(set, subsets)]
        # Number of subsets
        n_subsets = len(subsets)

        # Set venn objects and defaults
        if n_subsets == 2:
            venn_obj= matplotlib_venn.venn2
            circ_obj = matplotlib_venn.venn2_circles
            if labels is None:
                labels = ("A","B")
        if n_subsets == 3:
            venn_obj = matplotlib_venn.venn3
            circ_obj = matplotlib_venn.venn3_circles
            if labels is None:
                labels = ("A","B","C")
        if n_subsets > 3:
            print(f"Too many subsets: {n_subsets}")
        labels = [*map(str, labels)]

        # Default colors
        if colors is None:
            colors = sns.color_palette(palette=palette, n_colors=n_subsets, desat=None)

        # Create venn diagram
        venn = venn_obj(subsets=subsets, set_labels=labels, set_colors=colors, ax=ax, alpha=alpha, normalize_to=normalize_to)
        circ = circ_obj(subsets=subsets, **_circ_kws)

        # Title
        if title:
            ax.set_title(title, **_title_kws)

        for text in venn.set_labels:
            if text is not None:
                text.set_fontsize(fontsize_labels)
                text.set_fontweight("bold")

        for text in venn.subset_labels:
            if text is not None:
                text.set_fontsize(fontsize_numbers)

        return fig, ax, (venn, circ)

# Plot waterfall-style bar chart
def plot_waterfall(data:pd.Series,
                   title=None,
                   color=None,
                   cmap=plt.cm.seismic_r,
                   linewidth=1.618,
                   alpha=1,
                   edgecolor="black",
                   show_labels=True,
                   metric_label=None,
                   observation_label=None,
                   style="seaborn-white",
                   engine="matplotlib",
                   figsize=None,
                   ax=None,
                   metric_tick_fontsize=15,
                   observation_tick_fontsize=15,
                   rotation="auto",
                   orientation="vertical",
                   ascending=None,
                   errorbars=None,
                   label_kws=dict(),
                   title_kws=dict(),
                   bar_kws=dict(),
                   error_kws=dict(),
                   vmin=None,
                   vmax=None,
                   divergent="infer",
                   midpoint=0,
                   check_null=True,
                  ):
    orientation = orientation.lower().strip()
    assert engine in {"matplotlib", "pandas"}, "`engine` must be either matplotlib or pandas"
    if check_null:
        assert data.isnull().sum() == 0, "Please remove null values and try again"
    # Orientation
    if figsize is None:
        figsize = {"horizontal":(10,6.18033), "vertical":(6.18033,10)}[orientation]
    if rotation == "auto":
        rotation = {"horizontal":90, "vertical":0}[orientation]
    if ascending is not None:
        data = data.sort_values(ascending=ascending)
    # Keywrods
    _bar_kws = {"alpha":alpha,"linewidth":linewidth, "edgecolor":edgecolor }
    _bar_kws.update(bar_kws)
    _error_kws = {"color":"black", "fmt":"none"}
    _error_kws.update(error_kws)
    _label_kws = {"fontsize":18}
    _label_kws.update(label_kws)

    # Limits
    if divergent == "infer":
        divergent = all([data.min() < 0, data.max() > 0])

    if divergent:
        if vmax is None:
            vmax = data.abs().max()
        if vmin is None:
            vmin = -vmax
    else:
        if vmax is None:
            vmax = data.max()
        if vmin is None:
            vmin = data.min()
    if color is None:
        color = scalarmapping_from_data(data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=_bar_kws["alpha"], mode=1)

    # Data
    if errorbars is not None:
        errorbars = errorbars[data.index]
    x = np.arange(data.size)
    y = data.values

    # Plotting
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        if orientation == "horizontal":

            if engine == "pandas":
                data.plot(kind="bar",
                               color=color,
                               ax=ax,
                              **_bar_kws,
                         )
            if engine == "matplotlib":
                ax.bar(x, y, color=color, **_bar_kws)
                if midpoint is not None:
                    ax.axhline(midpoint,color="black")
            if errorbars is not None:
                ax.errorbar(x,y,yerr=errorbars, **_error_kws)

            _ = ax.set_xlim((-0.5, data.size-0.25))
            _ = ax.set_xticks(np.arange(data.size))
            _ = ax.set_xticklabels(data.index, rotation=rotation, fontsize=observation_tick_fontsize)
            _ = ax.set_yticklabels(ax.get_yticks(), fontsize=metric_tick_fontsize)
            if show_labels == False:
                ax.set_xticklabels([])
        if orientation == "vertical":
            if engine == "pandas":
                data.plot(kind="barh",
                               alpha=alpha,
                               color=color,
                               ax=ax)
            if engine == "matplotlib":
                ax.barh(x, y, color=color,  **_bar_kws)
            if errorbars is not None:
                ax.errorbar(y,x,xerr=errorbars, **_error_kws)
                if midpoint is not None:
                    ax.axvline(midpoint,color="black")
            _ = ax.set_ylim((-0.5, data.size-0.25))
            _ = ax.set_yticks(np.arange(data.size))
            _ = ax.set_yticklabels(data.index, rotation=rotation, fontsize=observation_tick_fontsize)
            _ = ax.set_xticklabels(ax.get_xticks(), fontsize=metric_tick_fontsize)
            if show_labels == False:
                ax.set_yticklabels([])

        # Labels
        if observation_label:
            if orientation == "vertical":
                ax.set_ylabel(observation_label, fontsize=18)
            if orientation == "horizontal":
                ax.set_xlabel(observation_label, fontsize=18)
        if metric_label:
            if orientation == "vertical":
                ax.set_xlabel(metric_label, fontsize=18)
            if orientation == "horizontal":
                ax.set_ylabel(metric_label, fontsize=18)
        if title:
            ax.set_title(str(title), fontsize=18, fontweight="bold")
        return fig,ax

# Adding annotations onto plots
def plot_annotation(labels, x:pd.Series, y:pd.Series, ax, adjust_label_positions=True,  **kwargs):
    """
    Dependency: https://github.com/Phlya/adjustText
    * Need to get the **kwargs and *args
    """
    def _get_text_objects(labels, x, y, ax, **kwargs):
        text_objects = []
        for label, x_i, y_i in zip(labels,x,y):
            text_objects.append(ax.text(x_i, y_i, label,  **kwargs))
        return text_objects
    x = pd.Series(x)
    y = pd.Series(y)
    assert set(labels) <= set(x.index), "All `labels` must be in `x.index"
    assert set(labels) <= set(y.index), "All `labels` must be in `y.index"
    x = x[labels]
    y = y[labels]
    text_objects = _get_text_objects(labels, x, y, ax, **kwargs)
    if adjust_label_positions:
        adjust_text(text_objects, x, y, ax=ax)
    return text_objects

# def plot_annotation(labels, x:pd.Series, y:pd.Series, ax, adjust_label_positions=True,  x_pad=0, y_pad=0, **kwargs): #! Need to test this with other functions
#     """
#     Dependency: https://github.com/Phlya/adjustText
#     * Need to get the **kwargs and *args
#     """
#     def _get_text_objects(labels, x, y, ax, **kwargs):
#         text_objects = []
#         for ((id, label), x_i, y_i) in zip(labels.iteritems(),x,y):
#             label = label.values[0]
#             text_objects.append(ax.text(x_i, y_i, label,  **kwargs))
#         return text_objects
#     if not is_dict_like(labels):
#         labels = dict(zip(labels, labels))
#     labels = pd.Series(labels)
#     x = pd.Series(x) + x_pad
#     y = pd.Series(y) + y_pad
#     assert set(labels.index) <= set(x.index), "All `labels` must be in `x.index"
#     assert set(labels.index) <= set(y.index), "All `labels` must be in `y.index"
#     x = x[labels.index]
#     y = y[labels.index]
#     text_objects = _get_text_objects(labels, x, y, ax, **kwargs)
#     if adjust_label_positions:
#         adjust_text(text_objects, x, y, ax=ax)
#     return text_objects

# Volcano plot
def plot_volcano(diffusion_values:pd.Series, test_values:pd.Series, tol_diffusion=0, tol_test=0.05, test_field="FDR", xlabel="log$_2$(FC)", ylabel="-log$_2$(FDR)", horizontal=None, linestyle=":", size_significant=50, size_nonsignificant=18, ax=None, figsize=(5,5), cmap=plt.cm.seismic_r, color_nonsignificant="#f8f8ff", title=None, alpha_ratio=0.618, linewidth=0.618,  edgecolor="black", style="seaborn-white", fill_nonsignificant="gray", alpha_fill=0.1618, show_annotations=False, adjust_annotation_positions=True, annot_kws=dict(),  *args):
    """
    # Generalized to all types of metrics (not just fold-change)
    # Added `test_field` (2018-May-11)
    Need to make sure this works correctly with MultiIndex
    Make a general use of tol_test so it makes sense to use tol_pvalue or something
    """
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        # Keywords
        _annot_kws = {}
        _annot_kws.update(annot_kws)

        # Vectors
        x = diffusion_values.copy()
        y = test_values.copy()
        assert np.all(x.index == y.index), "diffusion_values and test_values must have the same index"
        assert x.index.value_counts().max() == 1, "Cannot have duplicate values in the index"

        # DEGs
        significant_diffusion = x.abs() >= tol_diffusion
        significant_test = y < tol_test
        idx_degs = (significant_diffusion.astype(int) + significant_test.astype(int))[lambda z: z == 2].index
        vmax = x[idx_degs].abs().max()

        # Plot data
        c = map_colors(diffusion_values, mode=3, cmap=cmap, vmin=-vmax, vmax=vmax, format="hex")
        nonsignificant_labels = pd.Index(list(set(x.index) - set(idx_degs)))
        if color_nonsignificant is not None:
            mask = c.index.map(lambda label: label in nonsignificant_labels).values.astype(bool)
            c[mask] = color_nonsignificant
        s = pd.Series(x.index.map(lambda z:{True:50, False:18}[ z in idx_degs]), index=x.index)

        # Horizontal Lines

        if horizontal is None:
            horizontal = [tol_test]

        cond_A = hasattr(horizontal, "__len__") == True
        cond_B = type(linestyle) == str
        if all([cond_A, cond_B]):
            linestyle = [linestyle]*len(horizontal)

        # Alphas
        if len(horizontal) > 1:
            alphas = [1] + ((np.array([1]*(len(horizontal) - 1))*alpha_ratio).cumsum()).tolist()
        else:
            alphas = [1]

        # Synthesize Plot
        df_plot = pd.DataFrame([x,-np.log2(y.astype(float))], index=["x","y"]).T
        fig, ax, im = plot_scatter(data=df_plot, x="x", y="y", c=c, s=s, xlabel=xlabel, ylabel=ylabel, ax=ax, title=title, linewidth=linewidth, edgecolor=edgecolor, *args)

        # Synth Horizontal
        for i in range(len(horizontal)):
            ax.axhline(-np.log2(horizontal[i]), color="black", linestyle=linestyle[i], alpha=alphas[i])

        # Synth Vertical
        ax.axvline(-tol_diffusion, color="black", linestyle=linestyle[0])
        ax.axvline(tol_diffusion, color="black", linestyle=linestyle[0])

        # Fill nonsignficant area
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if fill_nonsignificant is not None:
            ax.fill_between([-tol_diffusion, tol_diffusion], *ylim, color=fill_nonsignificant, alpha=alpha_fill)
            ax.fill_betweenx( [min(ylim), -np.log2(tol_test)], *xlim, color=fill_nonsignificant, alpha=alpha_fill)
            ax.set(xlim=xlim, ylim=ylim)

        # Labels
        cond_C = show_annotations is not False
        cond_D = show_annotations is not None
        if all([cond_C, cond_D]):
            if show_annotations is True:
                show_annotations = idx_degs
            if  isinstance(show_annotations, (str, int)):
                show_annotations = [show_annotations]
            plot_annotation(labels=show_annotations, x=df_plot["x"], y=df_plot["y"], ax=ax, adjust_label_positions=adjust_annotation_positions, **_annot_kws)
        return fig, ax, im

# Plot prevalence of attributes in compositional data
def plot_prevalence(X:pd.DataFrame, color="teal", edgecolor="black", interval_type="closed", figsize=(13,5),  ax=None, style="seaborn-white", show_prevalence=[ 0.382, 0.5, 0.618, 1.0], show_ygrid=True, show_xgrid=False, show_legend=True, cmap=plt.cm.Dark2_r, attr_type="attr", obsv_type = "obsv", title=None, fill=True, title_kws=dict(), legend_kws=dict(), legend_title="Prevalence", legend_title_kws=dict(), n_xticks=25, yscale="linear", xlabel="auto", ylabel="auto", scale=False):

    _title_kws = {"fontsize":18, "fontweight":"bold"}
    _title_kws.update(title_kws)

    _legend_kws = { 'edgecolor': 'black', 'facecolor': 'white', 'fontsize': 15, 'frameon': True, "title_fontsize":15}
    _legend_kws.update(legend_kws)

    _legend_title_kws = {"size":15, "weight":"bold"}
    _legend_title_kws.update(legend_title_kws)


    with plt.style.context(style):

        if ax is  None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        # Dimensions
        n,m = X.shape

        # Prevalence
        x_prevalence = prevalence(X=X, interval_type=interval_type, name=None, scale=scale)

        # Plot
        ax.plot(x_prevalence.index, x_prevalence, color="gray")
        ax.scatter(x_prevalence.index, x_prevalence, color=color, edgecolor=edgecolor, linewidth=1, alpha=0.95)

        # Specific prevalence values
        if show_prevalence is not None:
            if any(map(lambda x: isinstance(x,float), show_prevalence)):
                show_prevalence = np.asarray(show_prevalence)
                assert all([np.all(show_prevalence <= 1), np.all(show_prevalence > 0)]), "If float values are provided then they represent ratios (i.e., Interval = (0,1] ])"
                show_prevalence = (show_prevalence * n).astype(int)

            colors = Chromatic.from_continuous(y=x_prevalence, cmap=cmap, vmin=0, vmax=max(show_prevalence))
            label_color = OrderedDict()
            for x_pos in sorted(show_prevalence):
                y_pos = x_prevalence.loc[x_pos]
                color = colors.obsv_colors[x_pos]
                # label = "N$_{%s}$ = %d\tN$_{%s}$ = %d"%(obsv_type, x_pos, attr_type, y_pos)
                label = "N$_{%s}$ = %d\tN$_{%s}$ = %d"%(obsv_type.replace("_","\_"), x_pos, attr_type.replace("_","\_"), y_pos) # 2019-04-19

                im = ax.plot([x_pos,x_pos], [0,y_pos], color=color, linestyle="-", linewidth=1, alpha=1, label=label)
                im = ax.plot([0 ,x_pos], [y_pos,y_pos], color=color, linestyle="-", linewidth=1, alpha=1)
                label_color[label] = color
                if fill:
                    im = ax.fill_between([0,x_pos], [y_pos,y_pos], color=color, alpha=0.1618)
            if show_legend:
                ax.legend(*format_mpl_legend_handles(label_color), **_legend_kws)
                if legend_title is not None:
                    ax.legend_.set_title(legend_title, prop=_legend_title_kws)

        # Set limits
        ax.set_ylim((0,x_prevalence.max()))
        ax.set_xlim((0,n))
        ax.set_xticks(np.linspace(0,n,min(n_xticks,n)).astype(int) )

        # Axis Labels
        if xlabel == "auto":
            if not scale:
                xlabel = f"Number of {obsv_type}s with $y_j$ {attr_type}s"
            else:
                xlabel = f"Ratio of {obsv_type}s with $y_j$ {attr_type}s"
        if ylabel == "auto":
            ylabel = f"Number of {attr_type}s with $x_i$ {obsv_type}s"
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)

        # Show grids
        ax.xaxis.grid(show_xgrid)
        ax.yaxis.grid(show_ygrid)

        # Title
        if title is not None:
            ax.set_title(title, **_title_kws)
        return fig, ax

# Plot compositional data
def plot_compositional(
    X:pd.DataFrame,
    y:pd.Series=None,
    c:pd.Series=None,
    color_density="black",
    edgecolor="gray",
    s=None,
    class_colors=None,
    attr_type="attr",
    unit_type="read",
    figsize=(13,8),
    title=None,
    style="seaborn-white",
    palette="hls",
    vertical_lines = np.mean,
    horizontal_lines=np.mean,
    cmap=plt.cm.gist_heat_r,
    scatter_kws=dict(),
    show_xgrid=False,
    show_ygrid=True,
    show_density_2d=True,
    show_density_1d=True,
    show_legend=True,
    legend_kws=dict(),
    legend_title=None,
    title_kws=dict(),
    legend_title_kws=dict(),
    axis_label_kws=dict(),
    line_kws=dict(),
    kde_1d_kws=dict(),
    kde_2d_kws=dict(),
    panel_pad=0.1,
    panel_size=0.618,
    ax=None,
    shade_lowest_2d=False,
    split_class_density=True,
    background_color="white",
    show_annotations=False,
    adjust_annotation_positions=True,
    annot_kws=dict(),
    log_depth=True,
    log_richness=False,
    ):
    def _get_line_data(y, data, lines, class_colors, _line_kws):
        kws_collection = list()
        if is_function(lines):
            if class_colors is not None:
                placeholder = list()
                for id_class, idx_query in pd_series_collapse(y).iteritems():
                    v = lines(data[idx_query])
                    placeholder.append(v)
                    kws = {**_line_kws}
                    kws["color"] = class_colors[id_class]
                    kws_collection.append(kws)
                lines = placeholder
            else:
                lines = lines(data)
        if is_number(lines):
            lines = [lines]
        if len(kws_collection) == 0:
            kws_collection = [_line_kws]*len(lines)
        return lines, kws_collection



    # Data
    X = X.fillna(0)
    # RIchness
    richness = (X > 0).sum(axis=1)
    zeromask_richness = richness == 0
    if np.any(zeromask_richness):
        warnings.warn("Removing the following observations because richness = 0: {}".format(zeromask_richness.index[zeromask_richness]))
        richness = richness[~zeromask_richness]
    if log_richness:
        richness = np.log10(richness)
    # Depth
    depth = X.sum(axis=1)
    zeromask_depth = depth == 0
    if np.any(zeromask_depth):
        warnings.warn("Removing the following observations because depth = 0: {}".format(zeromask_depth.index[zeromask_depth]))
        depth = depth[~zeromask_depth]
    if log_depth:
        depth = np.log10(depth)
    # Overlap
    index = pd.Index(sorted(depth.index & richness.index))
    depth = depth[index]
    richness = richness[index]
    X = X.loc[index,:]

    # Defaults
    _title_kws = {"fontsize":15, "fontweight":"bold"}
    _title_kws.update(title_kws)
    _legend_kws = {'fontsize': 12, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black'}#, 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    _legend_kws.update(legend_kws)
    _legend_title_kws = {"size":15, "weight":"bold"}
    _legend_title_kws.update(legend_title_kws)
    _axis_label_kws = {"fontsize":15}
    _axis_label_kws.update(axis_label_kws)
    _kde_1d_kws = {"rug":max(X.shape) < 5000, "hist_kws":{"alpha":0.382}, "kde_kws":{"alpha":0.618}} # Rug takes too long when theres a lot points
    _kde_1d_kws.update(kde_1d_kws)
    _kde_2d_kws = {"shade":True, "shade_lowest":shade_lowest_2d, "alpha":0.618, "linewidth":1}
    _kde_2d_kws.update(kde_2d_kws)
    _line_kws = {"color":color_density, "linewidth":1.618, "linestyle":":", "alpha":1.0}
    _line_kws.update(line_kws)
    _annot_kws = {}
    _annot_kws.update(annot_kws)
    _scatter_kws={"s":s, "cmap":cmap, "vmin":0}
    _scatter_kws.update(scatter_kws)
    # Colors
    use_entropy = True
    if y is not None:
        assert set(y.index) >= set(X.index), "All elements of X.index must be in y.index"
        y = y[X.index]
        if class_colors is not None:
            obsv_colors = y.map(lambda id_obsv:class_colors[id_obsv])
        else:
            colors = Chromatic.from_classes(y, palette=palette)
            obsv_colors = colors.obsv_colors[index]
            class_colors = colors.class_colors
        use_entropy = False
    else:
        show_legend=False

    if use_entropy:
        c = X.apply(lambda x:entropy(x, base=2), axis=1)

    if c is not None:
        c = pd.Series(c)
        assert set(c.index) >= set(X.index), "Not all observations from `X` are in `c`"
        c = c[index]
        obsv_colors = c
        colors_global = c
    else:
        colors_global = color_density


    axes = list()
    with plt.style.context(style):
        # Set up axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        axes.append(ax)

        # Scatter plot
        if y is not None:
            for id_class, idx_query in pd_series_collapse(y).iteritems():
                ax.scatter(depth[idx_query], richness[idx_query], c=obsv_colors[idx_query], edgecolor=edgecolor, label=id_class, **_scatter_kws)
        else:
            ax.scatter(depth, richness, c=colors_global, edgecolor=edgecolor, **_scatter_kws)
        # Limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Vertical lines
        if vertical_lines is not None:
            vertical_lines, kws_collection = _get_line_data(y, depth, vertical_lines, class_colors, _line_kws)
            for i, pos in enumerate(vertical_lines):
                ax.axvline(pos, **kws_collection[i])
        # Horizontal lines
        if horizontal_lines is not None:
            horizontal_lines, kws_collection = _get_line_data(y, richness, horizontal_lines, class_colors, _line_kws)
            for i, pos in enumerate(horizontal_lines):
                ax.axhline(pos, **kws_collection[i])
        ax.set(xlim=xlim, ylim=ylim)
        # Density (1D)
        if show_density_1d:
            # Setup
            divider = make_axes_locatable(ax)
            # Richness
            ax_right = divider.append_axes("right", pad=panel_pad, size=panel_size)
            ax_top = divider.append_axes("top", pad=panel_pad, size=panel_size)

            if split_class_density:
                if class_colors is None:
                    sns.distplot(richness, color=color_density, vertical=True, ax=ax_right, **_kde_1d_kws)
                    sns.distplot(depth, color=color_density, vertical=False, ax=ax_top, **_kde_1d_kws)
                else:
                    for id_class, idx_query in pd_series_collapse(y).iteritems():
                        sns.distplot(richness[idx_query], color=class_colors[id_class], vertical=True, ax=ax_right, **_kde_1d_kws)
                        sns.distplot(depth[idx_query], color=class_colors[id_class], vertical=False, ax=ax_top, **_kde_1d_kws)
            else:
                sns.distplot(richness, color=color_density, vertical=True, ax=ax_right, **_kde_1d_kws)
                sns.distplot(depth, color=color_density, vertical=False, ax=ax_top, **_kde_1d_kws)
            ax_right.set(ylim=ylim, xticklabels=[],yticklabels=[],yticks=ax.get_yticks())
            ax_top.set(xlim=xlim, xticklabels=[],yticklabels=[],xticks=ax.get_xticks())
            axes.append(ax_right)
            axes.append(ax_top)

        # Density (2D)
        if show_density_2d:
            if split_class_density:
                if class_colors is None:
                    sns.kdeplot(data=depth, data2=richness, color=color_density, zorder=0,  ax=ax, **_kde_2d_kws)
                else:
                    for id_class, idx_query in pd_series_collapse(y).iteritems():
                        try:
                            sns.kdeplot(data=depth[idx_query], data2=richness[idx_query],  color=class_colors[id_class], zorder=0, ax=ax, **_kde_2d_kws)
                        except ValueError:
                            warnings.warn("Could not compute the 2-dimensional KDE plot for the following class: {}".format(id_class))
            else:
                sns.kdeplot(data=depth, data2=richness, color=color_density, zorder=0,  ax=ax, **_kde_2d_kws)

        # Annotations
        conditions = [
            show_annotations is not False,
            show_annotations is not None
        ]
        if all(conditions):
            if show_annotations is True:
                show_annotations = X.index
            if  isinstance(show_annotations, (str, int)):
                show_annotations = [show_annotations]
            plot_annotation(labels=show_annotations, x=depth[show_annotations], y=richness[show_annotations], ax=ax, adjust_label_positions=adjust_annotation_positions, **_annot_kws)

        # Labels
        xlabel = "Depth [$N_{%s}$]"%(f"{unit_type}s")
        if log_depth:
            xlabel = "log$_{10}$(%s)"%(xlabel)
        ylabel = "Richness [$N_{%s}$]"%(f"{attr_type}s")
        if log_richness:
            ylabel = "log$_{10}$(%s)"%(ylabel)

        ax.set_xlabel(xlabel, **_axis_label_kws)
        ax.set_ylabel(ylabel, **_axis_label_kws)
        ax.xaxis.grid(show_xgrid)
        ax.yaxis.grid(show_ygrid)

        # Legend
        if show_legend:
            ax.legend(**_legend_kws)
            if bool(legend_title):
                ax.legend_.set_title(legend_title, prop=_legend_title_kws)
        # Title
        if title is not None:
            axes[-1].set_title(title, **_title_kws)

        # Background color
        for ax_query in axes:
            ax_query.set_facecolor(background_color)

        return fig, axes

# ===================
# Plot.ly Wrappers
# ===================
# ===================
# Bokeh Wrappers
# ===================
