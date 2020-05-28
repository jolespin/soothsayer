# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, logging, uuid, pathlib, posix, gzip, bz2, zipfile, subprocess, requests, operator
from collections import OrderedDict, defaultdict, Mapping
from io import TextIOWrapper

# PyData
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

## Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, hex2color, to_rgb, to_rgba #, ListedColormap
from palettable.cartocolors.diverging import TealRose_3
from palettable.cmocean.sequential import Tempo_8

# Datasets
from sklearn.datasets import load_iris

# Soothsayer
import soothsayer_utils as syu

functions_from_soothsayer_utils = [
'assert_acceptable_arguments', 'boolean', 'consecutive_replace', 'contains', 'dict_build', 'dict_collapse', 'dict_expand', 'dict_fill', 'dict_filter', 'dict_reverse', 'dict_tree',
'flatten', 'format_duration', 'format_header', 'format_path', 'fragment',  'get_timestamp', 'get_unique_identifier', 'hash_kmer', 'infer_compression', 'is_all_same_type',
'is_dict', 'is_dict_like', 'is_file_like', 'is_function', 'is_in_namespace', 'is_nonstring_iterable', 'is_number', 'is_path_like', 'is_query_class', 'iterable_depth', 'join_as_strings',
'pad_left', 'pv', 'range_like',  'reverse_complement', 'to_precision', "check_packages", "Suppress",
]

__all__ = {
    # Tree
    "infer_tree_type", "check_polytomy", "is_leaf","name_tree_nodes","prune_tree",
    # Need to organize these
    "pd_prepend_level_to_index", 'pd_series_to_groupby_to_dataframe', 'pd_dropduplicates_index', 'generate_random_sequence', 'get_iris_data', 'is_color', 'add_cbar_from_data', 'get_coords_contour', 'get_repr', 'get_parameters_ellipse', 'DIVERGING_KWS', 'determine_mode_for_logfiles', 'create_logfile', 'is_symmetrical', 'pd_dataframe_matmul', 'dataframe_to_matrixstring', 'infer_cmap', 'pd_dataframe_extend_index', 'configure_scatter', 'CMAP_DIVERGING', 'pd_series_collapse', 'is_graph', 'format_filename', 'pd_series_filter', 'map_colors', 'rgb_to_rgba', 'LEGEND_KWS', 'force_symmetry', 'is_rgb_like', 'COLOR_POSITIVE', 'Chromatic', 'COLOR_NEGATIVE', 'infer_vmin_vmax', 'pd_dataframe_query', 'get_coords_centroid', 'filter_compositional', 'scalarmapping_from_data', 'infer_continuous_type', 
    'format_mpl_legend_handles',
    }

for function_name in functions_from_soothsayer_utils:
    globals()[function_name] = getattr(syu, function_name)
    __all__.add(function_name)
__all__ = sorted(__all__)


# =============
# Defaults
# =============
LEGEND_KWS = {'fontsize': 15, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
DIVERGING_KWS = dict(h_neg=220, h_pos=15, sep=20, s=90, l=50)
CMAP_DIVERGING = sns.diverging_palette(**DIVERGING_KWS, as_cmap=True)
COLOR_NEGATIVE, COLOR_POSITIVE = sns.diverging_palette(**DIVERGING_KWS, n=2).as_hex()

# ===========
# # Types
# # ===========
def is_rgb_like(c):
    condition_1 = type(c) != str
    condition_2 = len(c) in [3,4]
    return all([condition_1, condition_2])

def is_color(obj):
    # Note: This can't handle values that are RGB in (0-255) only (0,1)
    try:
        to_rgb(obj)
        return True
    except ValueError:
        verdict = False
        if is_nonstring_iterable(obj):
            if all(type(x) in [float, int] for x in obj):
                if all(0 <= x <= 255 for x in obj):
                    verdict = True
        return verdict
def is_graph(obj):
    return hasattr(obj, "has_edge")

def is_number(x, num_type = np.number):
    return np.issubdtype(type(x), num_type)

# =============
# Checks
# =============
def is_symmetrical(X, tol=None):
    if X.shape[0] == X.shape[1]:
        if tol is None:
            return np.all(np.tril(X) == np.triu(X).T)
        if tol:
            return (np.tril(X) - np.triu(X).T).ravel().min() < tol
    else:
        return False

def force_symmetry(X):
    return (X + X.T)/2

# =============
# Tree
# =============


# Tree
def infer_tree_type(tree):
    tree_type = None
    query_type = str(tree.__class__).split("'")[1].split(".")[0]
    if query_type in {"skbio"}:
        tree_type = "skbio"
    if query_type in {"ete3"}:
        tree_type = "ete"
    assert tree_type is not None, "Please use either skbio or ete3 tree.  Tree type deterined as {}".format(query_type)
    return tree_type

# Is node a leaf?
def is_leaf(node, tree_type="infer"):
    assert_acceptable_arguments(tree_type, {"ete", "infer", "skbio"})
    if tree_type == "infer":
        tree_type = infer_tree_type(node)
    function_name = {"skbio":"is_tip", "ete":"is_leaf"}[tree_type]
    return getattr(node, function_name)()

# Check polytomy
def check_polytomy(tree, tree_type="infer"):
    assert_acceptable_arguments(tree_type, {"ete",  "infer", "skbio"})
    if tree_type == "infer":
        tree_type = infer_tree_type(tree)
        
    if tree_type == "ete":
        # Check bifurcation
        n_internal_nodes = len(list(filter(lambda node:node.is_leaf() == False, tree.traverse())))
        n_leaves = len(list(filter(lambda node:node.is_leaf(), tree.traverse())))
        if n_internal_nodes < (n_leaves - 1):
            raise Exception("Please resolve tree polytomy and force bifurcation: Use `tree.resolve_polytomy()` before naming nodes for `ete`")

    if tree_type == "skbio":
        # Check bifurcation
        n_internal_nodes = len(list(filter(lambda node:node.is_tip() == False, tree.traverse())))
        n_leaves = len(list(filter(lambda node:node.is_tip(), tree.traverse())))
        if n_internal_nodes < (n_leaves - 1):
            raise Exception("Please resolve tree polytomy and force bifurcation: Use `tree.bifurcate()` before naming nodes for `skbio`")


# Name tree nodes
def name_tree_nodes(tree,  node_prefix:str="y", tree_type:str="infer", polytomy_ok=False, inplace=False):
    """
    tree can be skbio or any ete3 tree
    """
    assert_acceptable_arguments(tree_type, {"ete", "infer", "skbio"})

    # In place or on a copy
    if not inplace:
        tree = tree.copy()
    # Determine tree type
    if tree_type == "infer":
        tree_type = infer_tree_type(tree)

    # Polytomy
    if not polytomy_ok:
        check_polytomy(tree=tree, tree_type=tree_type)
    # Name tree nodes
    intermediate_node_index = 1
    for node in tree.traverse():
        if not is_leaf(node):
            node.name = f"{node_prefix}{intermediate_node_index}"
            intermediate_node_index += 1
    if not inplace:
        return tree
    
# Prune tree
def prune_tree(tree,  leaves, tree_type="infer"):
    assert_acceptable_arguments(tree_type, {"ete",  "infer", "skbio"})
    if tree_type == "infer":
        tree_type = infer_tree_type(tree)
    if tree_type == "ete":
        tree.prune(leaves)
    if tree_type == "skbio":
        tree = tree.shear(leaves)
        tree.prune()
    return tree

# =============
# Miscellaneous
# =============
# Random sequence generation
def generate_random_sequence(size:int=100, alphabet=["A","T", "C", "G"], weights=[0.25,0.25,0.25,0.25], random_state=None):
    rs = np.random.RandomState(random_state)
    x = np.random.choice(alphabet, size=size, replace=True, p=weights)
    return "".join(x)

# For ete3 ClusterNode
def dataframe_to_matrixstring(df):
    return df.to_csv(None, sep="\t",index_label="#Names")


# =====
# Formatting
# =====
# Creates a unique identifier
def get_unique_identifier():
    return uuid.uuid4().hex
# Infer compression
def infer_compression(path:str):
    path = format_path(path)
    compression = None
    ext_zip = (".zip")
    ext_gzip = (".gz", ".gzip", ".pgz")
    ext_bz2 = (".bz2", ".bzip2", ".pbz2")
    if path.endswith(ext_gzip):
        compression= "gzip"
    if path.endswith(ext_bz2):
        compression = "bz2"
    if path.endswith(ext_zip):
        compression = "zip"
    return compression
# Format filename
def format_filename(name, replace="_"):
    Ar_name = np.array(list(name))
    idx_nonalnum = [*map(lambda x: x.isalnum() == False,  Ar_name)]
    Ar_name[idx_nonalnum] = replace
    return "".join(Ar_name)


# Get repr for custom classes
def get_repr(class_name, instance_name=None, *args):
    header = "{}(name = {})".format(class_name, instance_name)
    info = format_header(header)
    for field in args:
        info += "\n\t* {}".format(field)
    return info


# ======
# Matplotlib and colors
# ======
# Format handles for a matplotlib legend
def format_mpl_legend_handles(cdict_handles, label_specific_kws=None, marker="o", markeredgecolor="black", markeredgewidth=1,**args ):
    """
    More info on parameters: https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_marker
    Usage: plt.legend(*format_mpl_legend_handles(cdict_handles),
                      loc="lower center",
                      bbox_to_anchor=(0.5,-0.15),
                      fancybox=True, shadow=True,
                      prop={'size':15})

    Input: cdict_handles = Dictionary object of {label:color}
    Output: Tuple of handles and labels
    """
    cdict_handles = OrderedDict(cdict_handles)

    handle_kws = {"marker":marker, "markeredgecolor":markeredgecolor, "markeredgewidth":markeredgewidth, "linewidth":0}
    handle_kws.update(args)

    labels = list(cdict_handles.keys())
    if label_specific_kws is not None:
        label_specific_kws = dict(label_specific_kws)
        assert set(labels) == set(label_specific_kws.keys()), f"If `label_specific_kws` is not None then it must have all elements from `cdict_handles`"
    else:
        label_specific_kws = {label:handle_kws for label in labels}

    handles = list()
    for label, color in cdict_handles.items():
        handle = plt.Line2D([0,0],[0,0], color=color, **label_specific_kws[label])
        handles.append(handle)
    return (handles, labels)

# Convert RGB ==> RGBA
def rgb_to_rgba(u, alpha=1.0, missing_alpha=1.0, name=None, verbose=True):
    """
    `u` is a pd.Series of {label:(r,g,b)}
    `alpha` can be a pd.Series, dict, or scalar
    """
    # Convert to Series
    u = pd.Series(u)
    assert len(u.values[0]) == 3, "Not all values are (r,g,b)"
    if is_dict(alpha):
        alpha = pd.Series(alpha, name="alpha")

    if is_number(alpha):
        assert 0.0 <= alpha <= 1.0, "`alpha` should be between 0 and 1"
        alpha = pd.Series([float(alpha)]*u.size, index=u.index, name="alphas")
    # Missing alpha values
    alpha = alpha[u.index]
    if pd.isnull(alpha).sum() > 0:
        alpha[alpha.isnull()] = missing_alpha
        if verbose:
            print(f"Warning: Filling in missing alpha values as {missing_alpha}", file=sys.stderr)
    # Convert to RGBA
    return pd.Series(map(lambda x: (*u[x],alpha[x]), u.index), index=u.index, name=name)

# Map Colors
def map_colors(y:pd.Series,
               mode=0,
               alpha=1.0,
               format="rgba",
               outlier_value=-1,
               outlier_color="white",
               palette="hls",
               cmap=None,
               desat=None,
               vmin=None,
               vmax=None,
               missing_alpha=1.0,
               name=None,
               ):
    """
    type(u) should be pd.Series for mode=1 but mode=0 can be any iterable

    mode=0: maps `u` to `n` colors where `n` is number of elements [e.g. a list of `n` clusters]
    mode=1: maps `u` to `m` colors where `m` is number of unique elements [e.g. a target vector]
    mode=2: maps `u` dict of groups -> values [e.g. a dictionary of clusters]
    mode=3: maps `u` dict of values to continuous colormap [e.g. a dictionary]
    mode=4: returns tuple of cmap and lmap
    [hex, rgb, rgba]

    `cmap` only works with mode == 3
    """
    # Nested kws
    nested_kws = dict(outlier_value=outlier_value, outlier_color=outlier_color, palette=palette, desat=desat)

    # Convert to pd.Series
    y = pd.Series(y)

    # Expand {group:iterable} to series
    if mode == 2:
        y = dict_expand(y.to_dict(), into=pd.Series)

    # One-to-one color mapping
    if mode==0:
        d_class_color = OrderedDict(zip(sorted(y.tolist()), sns.color_palette(palette=palette, n_colors=y.size, desat=desat)))
        if outlier_value in d_class_color:
            d_class_color[outlier_value] = outlier_color

        # Convert to series and format output
        class_colors = pd.Series(d_class_color, name=name).sort_index()
        if format == "hex":
            return class_colors.map(rgb2hex)
        if format == "rgb":
            return class_colors
        if format == "rgba":
            return rgb_to_rgba(class_colors, alpha=alpha, missing_alpha=missing_alpha, verbose=False)

    # Vertical list color mapping
    if mode in {1,2}:
        class_colors = map_colors(y.unique(),  mode=0, alpha=alpha, format="rgb", **nested_kws)
        obsv_colors = y.map(lambda x: class_colors[x])
        obsv_colors.name = name
        if format == "rgb":
            return obsv_colors
        if format == "rgba":
            return rgb_to_rgba(obsv_colors, alpha=alpha, missing_alpha=missing_alpha, name=name, verbose=False)
        if format == "hex":
            return obsv_colors.map(rgb2hex)

    # Continuous data
    if mode==3:
        # Converging or diverging color palettes
        return scalarmapping_from_data(y, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, format=format)

    # Return color mapping for vertical list and legend
    if mode == 4:
        obsv_colors = map_colors(y, mode=1, alpha=1.0, format=format, **nested_kws)
        class_colors = map_colors(y.unique(), mode=0, alpha=alpha, format=format, **nested_kws)
        return (obsv_colors, class_colors)

# Determine continuous data type
def infer_continuous_type(data:pd.Series):
    assert np.all(data.map(is_number)), "All values in `data` must be numerical"

    _vmin = data.min()
    _vmax = data.max()

    # Conditions
    conditions = defaultdict(list)
    conditions["A"].append(_vmin < 0)
    conditions["A"].append(_vmax > 0)

    # +/-
    if all(conditions["A"]):
        return "diverging"
    # Other
    else:
        return "sequential"

# Determining vmin and vmax
def infer_vmin_vmax(data:pd.Series, continuous_type="infer"):
    vmin = None
    vmax = None
    # Infer continuous type
    if continuous_type in ["infer", None]:
        continuous_type = infer_continuous_type(data)
    # +/-
    if continuous_type == "diverging":
        vmax = data.abs().max()
        vmin = -vmax
    # Other
    if continuous_type == "sequential":
        vmax = data.max()
        vmin = data.min()
    assert all(map(bool, [vmin,vmax])), "`vmin` and `vmax` should not be None at this point.  Please check `infer_continuous_type`"
    return vmin, vmax

# Determing converging or diverging colormap
def infer_cmap(data:pd.Series, diverging=TealRose_3.mpl_colormap, sequential=Tempo_8.mpl_colormap):
    """
    diverging and sequential are both mpl colormaps or compatible
    """
    continuous_type = infer_continuous_type(data)
    return {"diverging":diverging, "sequential":sequential}[continuous_type]

# Mapping scalar values to colors
def scalarmapping_from_data(data:pd.Series, cmap=None, vmin=None, vmax=None, alpha=1.0, mode=1, format="rgba", name=None, missing_alpha=1.0):
    """
    Function that takes in numerical values and converts them to a pd.Series color mapping
    Must provide either `data` or (`cmap`, `vmin`, and `vmax`).  The latter is only compatible with mode == 0
    Input:
        type(data) is pd.Series
        cmap = mpl or sns cmap object
        vmin = minimumum value
        vmax = maximumum value
    mode = 0:
        Output: Fitted scalarmappable
    mode = 1:
        Output: Transformed values to colors
    Format:
        `rgba`, `rgb`, or `hex`
    Log:
        09.28.2017
    """
    # Useable data
    data = pd.Series(data).dropna()

    # Condtions
    conditions = defaultdict(list)

    # vmin/vmax
    continuous_type = infer_continuous_type(data)
    conditions["A"].append(vmin in ["infer", None])
    conditions["A"].append(vmax in ["infer", None])
    if any(conditions["A"]):
        assert all(conditions["A"]), "If `vmin` or `vmax` is `infer` then both must be `infer`.  Explicitly state  `vmin` and `vmax`"
        vmin, vmax = infer_vmin_vmax(data, continuous_type=continuous_type)

    # Colormap
    if cmap is None:
        cmap = infer_cmap(data)

    # Scalar Mapper
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # Outputs
    if mode == 0:
        return sm
    if mode == 1:
        obsv_colors = data.map(lambda x: sm.to_rgba(x)[:-1])
        if format == "rgb":
            return obsv_colors
        if format == "rgba":
            return rgb_to_rgba(obsv_colors, alpha=alpha, missing_alpha=missing_alpha, name=name, verbose=False)
        if format == "hex":
            return obsv_colors.map(rgb2hex)

# Add colorbar from the data
def add_cbar_from_data(fig, cmap, data=None, vmin=None, vmax=None, cbar_pos=[0.925, 0.1, 0.015, 0.8], label="infer", cbar_kws=dict(), tick_kws=dict(), **args):
    """
    Input:
        `fig` = a matplotlib figure
        `cmap` = matplotlib or seaborn cmap
        `data` = pd.Series of data
    Output:
        cbar, ax_colorbar
    Log:
        03.11.2019 Changed `u` to `data` and adjusted the style for conditions, now returns cbar, ax_colorbar
        09.28.2017 removed inactivate arguments, made fig, cmap, and u mandatory, added cbar_pos for customization in cbar position
    """
    conditions = defaultdict(list)
    for condition in ["A","B"]:
        conditions[condition].append(fig is not None)
        conditions[condition].append(cmap is not None)
    conditions["A"].append(data is not None)
    conditions["B"].append(vmin is not None)
    conditions["B"].append(vmax is not None)

    assert all(conditions["A"]) or all(conditions["B"]), "Must input either `fig`, `cmap`, and `data` arguments or `fig`, `cmap`, `vmin` and `vmax`"

    # ax_colorbar.tick_params
    _tick_kws = {}# {"size":15}
    _tick_kws.update(tick_kws)
    # cbar.set_label
    _cbar_kws = {"fontsize":18, "rotation":270, "labelpad":25}
    _cbar_kws.update(cbar_kws)
    # From data
    if all(conditions["A"]):
        if label == "infer":
            label = data.name

        sm = scalarmapping_from_data(data, cmap=cmap, vmin=vmin, vmax=vmax, mode=0)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

    # Set parameters
    ax_colorbar = fig.add_axes(cbar_pos)
    sm._A = []

    # Color bar
    cbar = fig.colorbar(sm, cax=ax_colorbar, **args)
    ax_colorbar.tick_params(**_tick_kws)
    # Labels
    if label:
        cbar.set_label(label, **_cbar_kws)
    return cbar, ax_colorbar

# Configure Scatter Plot Data
def configure_scatter(

                     # Main
                     data,
                     x=None,
                     y=None,
                     z=None,
                     c="gray",
                     s=None,
                     alpha=1.0,

                     # Aspects
                     cmap = None,
                     vmin=None,
                     vmax=None,
                     dimensions=2,

                     # Missing data
                     missing_color='#FFFFF0',
                     missing_alpha=1.0,
                     verbose=True,
                     ):

    """
    Preprocessing for 2-/3D scatter plot data.  Formats color, size, and alpha values.  Designed for matplotlib input but should be able to be universal for Plot.ly and Bokeh
    """
     # ==================================================
    # DataFrame
    # ==================================================
    # Default x and y axis
    if x is None:
        x = data.iloc[:,0]
    if y is None:
        y = data.iloc[:,1]
    if type(x) == str:
        x = data[x]
    if type(y) == str:
        y = data[y]
    if dimensions == 3:
        if z is None:
            z = data.iloc[:,2]
        if type(z) == str:
            z = data[z]
    # Coordinates
    if dimensions == 2:
        coords = pd.DataFrame([x,y], index=["x","y"]).T
    if dimensions == 3:
        coords = pd.DataFrame([x,y,z], index=["x","y","z"]).T

    # Defaults
    if c is None:
        c = "gray"
    if s is None:
        N = data.shape[0]
        s = np.square((1/np.log(N))*50)

    # Is color a labeled column?
    if type(c) == str:
        # Use values from color column
        if c in data.columns:
            c = data[c]
        # Color all values same color
        else:
            c = pd.Series([c]*len(x), index=data.index)

    # Convert colors to pd.Series
    c = pd.Series(c)[data.index]
    scalar_vector = None

    # Continuous mapping for colors
    if is_number(c.values[0]):
        if pd.isnull(c.values[0]) == False:
            scalar_vector = c.copy()
            if cmap is None:
                cmap = infer_cmap(c)
                c = scalarmapping_from_data(c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=1.0, format="rgb", mode=1, name=c.name)

    # Fill missing colors
    if pd.isnull(c).sum() > 0:
        c[c.isnull()] = missing_color
        if verbose:
            print(f"Warning: Filling in missing color values as {missing_color}", file=sys.stderr)

    # Set up alpha values
    # If Hex string, convert to RGB
    if isinstance(c.values[0], (str, bytes)):
        c = c.map(to_rgb)

    # If RGB or RGBA
    # WARNING: Cannot handle an instance where there is a subset of the indices and the values of the subset are floats/ints
    if alpha is not None:
        if hasattr(c.values[0], '__iter__'):
            # If RGB then add alpha to RGBA
            if len(c.values[0]) == 3:
                if is_number(alpha):
                    assert 0.0 <= alpha <= 1.0, "`alpha` should be between 0 and 1"
                    alpha = pd.Series([float(alpha)]*c.size, index=c.index, name="alphas")
                # Fill missing alpha values
                if pd.isnull(alpha).sum() > 0:
                    alpha[alpha.isnull()] = missing_alpha
                    if verbose:
                        print(f"Warning: Filling in missing alpha values as {missing_alpha}", file=sys.stderr)
                # Convert to RGBA
                c = pd.Series(map(lambda x: (*c[x],alpha[x]), c.index), index=c.index, name=c.name)
    # Size
    if s is not None:
        if type(s) == str:
            if s in data.columns:
                s = data[s]
        if is_number(s):
            s = pd.Series([s]*len(x), index=data.index, name="sizes")
        # Convert sizes to pd.Series
        s = pd.Series(s, name="sizes")[data.index]
        # Fill in missing sizes
        if pd.isnull(s).sum() > 0:
            med = s.dropna().median()
            s[s.isnull()] = med
            if verbose:
                print(f"Warning: Filling in missing size values as {med}", file=sys.stderr)
    return {
            "coords":coords,
            "colors":c,
            "sizes":s,
            "cmap":cmap,
            "scalar_vector":scalar_vector,
           }

# # Colors class object
# class Hue(str):
#     # Built in
#     def __new__(cls, color, name=None, **metadata):
#         assert is_color(color), "`{}` is not a color".format(color)
#         cls.color = rgb2hex(to_rgb(color))
#         cls.name = name
#         cls.metadata = metadata
#         return super(Hue, cls).__new__(cls, cls.color)
#     def __repr__(self):
#         return self.color
#     def __str__(self):
#         return self.color
#     # View
#     def view(self):
#         sns.palplot([self.color])
#     # Conversion
#     def to_hex(cls):
#         return cls.color
#     def to_rgb(cls, into=tuple):
#         return into(to_rgb(cls.color))
#     def to_rgba(cls, alpha=None, into=tuple):
#         return into(to_rgba(cls.color, alpha=alpha))
#     def to_rgb_255(cls, into=tuple):
#         return into(np.asarray(cls.to_rgb())*255)
#     def to_rgba_255(cls, alpha=None, into=tuple):
#         return into(np.asarray(cls.to_rgba(alpha=alpha))*255)

# Color collections
class Chromatic(object):
    """
    # =================
    # Continuous colors
    # =================
    colors_sepal_length = Chromatic.from_continuous(y=X_iris["sepal_length"], name="sepal_length", cmap=plt.cm.magma)
    print(colors_sepal_length.obsv_colors.iloc[:3])
    # iris_0    #deb5e8
    # iris_1    #efc5e6
    # iris_2    #f9d7e7
    # Name: sepal_length, dtype: object
    colors_sepal_length.__dict__.keys()
    # dict_keys(['name', '__synthesized_', 'vmin', 'vmax', 'cmap', 'continuous_type', 'format', 'mode', 'obsv_colors', 'y'])

    # ==============
    # Classes colors
    # ==============
    colors_species = Chromatic.from_classes(y_iris, name="iris species", palette="Set2")
    print(colors_species.class_colors)
    # setosa        #66c2a5
    # versicolor    #fc8d62
    # virginica     #8da0cb
    print(colors_species.obsv_colors.iloc[:3])
    # iris_0    #66c2a5
    # iris_1    #66c2a5
    # iris_2    #66c2a5
    # Name: Species, dtype: object
    colors_species.__dict__.keys()
    # dict_keys(['name', '__synthesized_', 'format', 'mode', 'obsv_colors', 'class_colors', 'y', 'palette', 'color_kws'])
    """
    def __init__(self, **kwargs):
        attributes = {"name":None, "__synthesized_":datetime.datetime.utcnow()}
        attributes.update(kwargs)
        for k,v in attributes.items():
            setattr(self, k,v)

    @classmethod
    def from_classes(cls, y:pd.Series, name=None, palette="hls", class_type=None, obsv_type=None, format="hex", outlier_value=-1, outlier_color='white', color_kws=dict()):
        if is_dict(y):
            y = pd.Series(y)
        obsv_colors, class_colors = map_colors(y, mode=4, format=format, palette=palette, outlier_value=outlier_value, outlier_color=outlier_color, **color_kws)
        attributes = { "name":name, "format":format, "mode":"classes", "obsv_colors":obsv_colors,"class_colors":class_colors, "y":y.copy(), "palette":palette,"color_kws":color_kws}
        return cls(**attributes)

    @classmethod
    def from_continuous(cls, y:pd.Series, name=None, cmap="infer", vmin="infer", vmax="infer", format="hex", **attrs):
        if is_dict(y):
            y = pd.Series(y)
        assert np.all(y.map(is_number)), "All values in `y` must be numerical"

        conditions = defaultdict(list)

        # vmin/vmax
        continuous_type = infer_continuous_type(y)
        conditions["A"].append(vmin in ["infer", None])
        conditions["A"].append(vmax in ["infer", None])
        if any(conditions["A"]):
            assert all(conditions["A"]), "If `vmin` or `vmax` is `infer` then both must be `infer`.  Explicitly state  `vmin` and `vmax`"
            vmin, vmax = infer_vmin_vmax(y, continuous_type=continuous_type)

        # cmap
        if cmap in ["infer", None]:
            cmap = infer_cmap(y)

        # Get colors
        obsv_colors = map_colors(y, mode=3, vmin=vmin, vmax=vmax, format=format, cmap=cmap)
        attributes = {"vmin":vmin, "vmax":vmax, "cmap":cmap, "continuous_type":continuous_type, "name":name, "format":format, "mode":"continuous", "obsv_colors":obsv_colors, "y":y.copy()}
        return cls(**attributes)

    def plot_colors(self, color_type="infer", size=1):
        accepted_colortypes = {"infer",None,"classes", "obsvs", "leaves", "obsvervations"}
        assert color_type in accepted_colortypes, f"`color_type` must be one of the following: {accepted_colortypes}"
        if color_type in ["infer",None]:
            if hasattr(self, "class_colors"):
                color_type = "classes"
            else:
                color_type = "obsvs"

        if color_type in ["obsvs", "observations", "leaves"]:
            if hasattr(self, "continuous_type"):

                colors = self.obsv_colors[self.y.sort_values().index]
            else:
                colors = self.obsv_colors.sort_values()
        if color_type == "classes":
            assert hasattr(self, "class_colors"), "There isn't a `class_colors` object.  Is this continuous data?  Use `color_type = 'obsvs' instead."
            colors = self.class_colors
        sns.palplot(colors, size=size)

# ===========
# Coordinates
# ===========
# Get 3d contour interpolation
def get_coords_contour(x,y,z,resolution = 50, method_iterpolation='linear'):
    # Adapted from @user10140465
    # https://stackoverflow.com/questions/52935143/how-to-do-a-contour-plot-from-x-y-z-coordinates-in-matplotlib-plt-contourf-or/52937196#52937196
    resolution = f"{resolution}j"
    X,Y = np.mgrid[min(x):max(x):complex(resolution),
                   min(y):max(y):complex(resolution)]
    points = list(zip(x,y))
    Z = griddata(points, z, (X, Y), method=method_iterpolation)
    return X,Y,Z

# Format centroids for group of indices
def get_coords_centroid(df_xy, groups, metric=np.mean, into=list, dimensions=2):
    """
    groups: {group_A: [obsv_1, obsv_2], group_B: [obsv_3, obsv_4]}
    Output:
    [(group_i, (x_i,y_i))]

    Note: This only works for 2D
    Previous names: centroids_2d
    """
    assert dimensions == 2, "Currently only is available for 2 dimensional datasets for plotting purposes"

    # Compute centroid for 2 vectors based on groups
    def _compute(df_xy, groups, metric):
        if is_nonstring_iterable(groups):
            groups = dict(groups)
        groups = {r:p for p,q in groups.items() for r in q}
        centroids_xy = df_xy.groupby(groups).apply(metric, axis=0)
        if isinstance(centroids_xy, pd.Series):
            centroids_xy = pd.DataFrame.from_items(zip(centroids_xy.index, centroids_xy.values)).T
            centroids_xy.columns = ["x","y"]
        return centroids_xy
    df_centroid_xy = _compute(df_xy=df_xy, groups=groups, metric=metric)

    if into == pd.DataFrame:
        return df_centroid_xy
    else:
        return into([*zip(df_centroid_xy.index, zip(df_centroid_xy.iloc[:,0], df_centroid_xy.iloc[:,1]))])

# Format ellipse attributes
def get_parameters_ellipse(df_xy, groups, metric=np.mean, n_std=3, into=pd.DataFrame):
    """
    groups: {group_A: [obsv_1, obsv_2], group_B: [obsv_3, obsv_4]}

    Previous names: centroids_ellipse
    """
    # Compute ellipse attributes
    def _compute(df_xy, idx, metric, n_std):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
        # Data
        x =  df_xy.iloc[:,0][idx].as_matrix()
        y =  df_xy.iloc[:,1][idx].as_matrix()
        # Reduce
        x_reduce = metric(x)
        y_reduce = metric(y)
        # Diff
        cov = np.cov(x, y)
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)

        return OrderedDict([
                          ("x", x_reduce),
                          ("y", y_reduce),
                          ("theta", theta),
                          ("width", width),
                          ("height", height),
                            ])
    ellipse_data = list()
    for group, idx in groups.items():
        ellipse_data.append(pd.Series(_compute(df_xy=df_xy, idx=idx, metric=metric, n_std=n_std), name=group))
    return into(ellipse_data)


# ===========
# Logging
# ===========
# Log files
def create_logfile(name, path,level=logging.INFO, mode="w"):
    """Function setup as many loggers as you want"""
    # https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    handler = logging.FileHandler(path, mode=mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Determine whether or not logfile should be overwritten or appended
def determine_mode_for_logfiles(path, force_overwrite):
    if force_overwrite:
        mode_for_logfiles = "w"
    else:
        if os.path.exists(path):
            mode_for_logfiles = "a"
        else:
            mode_for_logfiles = "w"
    return mode_for_logfiles


# ===========
# Pandas
# ===========
# Prepend level to pandas index
def pd_prepend_level_to_index(x, level_value, level_name=None):
    """
    Prepend a level to a pd.Series or pd.DataFrame
    
    Usage: 
    data = list("ATG")
    x = pd.Series(
        data, 
        index=pd.Index(range(0,len(data)), name="Position"),
        name="START",
    )
    print(x)
    # Position
    # 0    A
    # 1    T
    # 2    G
    # Name: START, dtype: object
    print(pd_prepend_level_to_index(x, "Chr1", level_name="Chromosome"))
    # Chromosome  Position
    # Chr1        0           A
    #             1           T
    #             2           G
    # Name: START, dtype: object
    
    Adapted from the following source:
    https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex
    """
    return pd.concat([x], keys=[level_value], names=[level_name])

# Collapse a pandas series
def pd_series_collapse(u:pd.Series, into=pd.Series, type_collection=set):
    """
    Input: Pandas Series object with redundant values (e.g. iris targets)
    Output: Object organized by group

    e.g., a:b to {b:[x,y,z]} or another pd.Series with a similar vibe
    """
    data = into(dict(map(lambda x: (x[0], type_collection(x[1].index)), u.groupby(u))))
    if hasattr(data, "name"):
        data.name = u.name
    return data

# Convert pandas series to a groupby object and then a dataframe of the groups
def pd_series_to_groupby_to_dataframe(data, by, fillna=None):
    df = pd.DataFrame([*map(lambda group: pd.Series(group[1], name=group[0]).reset_index(drop=True), data.groupby(by))]).T
    if fillna is not None:
        df = df.fillna(fillna)
    return df

# Filter pandas series
def pd_series_filter(func, u):
    mask = u.map(func)
    return u[mask]

# Query a pandas dataframe
def pd_dataframe_query(df:pd.DataFrame, conditions, mode="all"):
    """
    conditions = dict with key as column and value as the condition function

    conditions = {"run": lambda id_run: id_run in [3,4,5,9], "sample-category": lambda x: x in ["unknown", "control"]}
    pd_dataframe_query(df_meta, conditions)
    """
    number_of_conditions = len(conditions)
    assert number_of_conditions > 0, "Must provide at least one conditional"
    assert mode in {"any", "all"}, "`mode` must be either 'any' or 'all'"
    if mode == "all":
        tol = number_of_conditions
    if mode == "any":
        tol = 1

    masks = list()
    for key, condition in conditions.items():
        masks.append(df[key].map(condition))
    idx_query = pd.DataFrame(masks).sum(axis=0)[lambda x: x >= tol].index
    return df.loc[idx_query,:]

# Matrix Multiplication
def pd_dataframe_matmul(A:pd.DataFrame,B:pd.DataFrame):
    """
    A.shape (n,m)
    B.shape (m,p)
    Output: (n,p)
    """
    return pd.DataFrame(np.matmul(A, B.loc[A.columns,:]), index=A.index, columns=B.columns)

# Extend a DataFrame
def pd_dataframe_extend_index(index_extended, df=None, fill=np.nan, axis=0):
    """
    Extend the index of pd.DataFrame like pandas used to do before 23.4
    """
    if df is None:
        df = pd.DataFrame()
    if axis == 0:
        idx_extend = set(index_extended) - set(df.index)
        A = np.empty((len(idx_extend), df.shape[1]))
        A[:] = np.nan
        return pd.concat([df, pd.DataFrame(A, index=idx_extend, columns=df.columns)]).fillna(fill)
    if axis == 1:
        idx_extend = set(index_extended) - set(df.columns)
        A = np.empty(( df.shape[0], len(idx_extend)))
        A[:] = np.nan
        return pd.concat([df, pd.DataFrame(A, index=df.index, columns=idx_extend)]).fillna(fill)

# Drop duplicates index
def pd_dropduplicates_index(data, keep="first", axis=0):
    if axis in {0, None}:
        return data[~data.index.duplicated(keep=keep)]
    if axis == 1:
        data = data.T
        return data[~data.index.duplicated(keep=keep)].T

# # =======
# # Filters
# # =======
# Filter composition 2D data
def filter_compositional(
    X:pd.DataFrame,
    tol_depth=None,
    tol_prevalence=None,
    tol_richness=None,
    tol_count=None,
    order_of_operations=["depth", "prevalence", "richness","count"],
    mode="highpass",
    interval_type="closed",
    ):
    """
    Input: pd.DataFrame (rows=obsvs, cols=attrs)
    Output: pd.DataFrame filtered
    depth:  The minimum depth (sum per row) (axis=0)
    prevalence: The minimum number of observations that must contain the attribute (axis=1)
    count: The minimum number of total counts contained by the attribute (axis=1)
    richess: The minimum number of detected attributes (axis=0)
    """
    assert_acceptable_arguments(query=order_of_operations,target=["depth", "prevalence", "richness","count"], operation="le")
    assert_acceptable_arguments(query=[mode],target=["highpass", "lowpass"], operation="le")
    assert_acceptable_arguments(query=[interval_type],target=["closed", "open"], operation="le")


    def _get_elements(data,tol,operation):
        return data[lambda x: operation(x,tol)].index

    def _filter_depth(X, tol, operation):
        data = X.sum(axis=1)
        return X.loc[_get_elements(data, tol, operation),:]

    def _filter_prevalence(X, tol, operation):
        conditions = [
            isinstance(tol, float),
            0.0 < tol <= 1.0,
        ]

        if all(conditions):
            tol = round(X.shape[0]*tol)
        data = (X > 0).sum(axis=0)
        assert tol <= X.shape[0], "If prevalence is an integer ({}), it cannot be larger than the number of samples ({}) in the index".format(tol, X.shape[0])
        return X.loc[:,_get_elements(data, tol, operation)]

    def _filter_richness(X, tol, operation):
        data = (X > 0).sum(axis=1)
        return X.loc[_get_elements(data, tol, operation),:]

    def _filter_count(X, tol, operation):
        data = X.sum(axis=0)
        return X.loc[:,_get_elements(data, tol, operation)]
    if interval_type == "closed":
        operations = {"highpass":operator.ge, "lowpass":operator.le}
    if interval_type == "open":
        operations = {"highpass":operator.gt, "lowpass":operator.lt}

    # Defaults
    if tol_richness is None:
        tol_richness = 1
    if tol_count is None:
        tol_count = 1
    if tol_prevalence is None:
        tol_prevalence = 1

    functions = dict(zip(["depth", "prevalence", "richness","count"], [_filter_depth, _filter_prevalence, _filter_richness, _filter_count]))
    thresholds = dict(zip(["depth", "prevalence", "richness","count"], [tol_depth, tol_prevalence, tol_richness, tol_count]))


    for strategy in order_of_operations:
        tol = thresholds[strategy]
        if tol is not None:
            X = functions[strategy](X=X,tol=tol, operation=operations[mode])

    return X
# ===========
# Prototyping
# ===========
def get_iris_data(return_data=["X", "y", "colors"], noise=None, return_target_names=True, palette="Set2", desat=1, random_state=0):
    """
    return_data priority is [X, y, colors]
    """
    if isinstance(return_data, str):
        return_data = [return_data]
    assert set(return_data) <= set(["X", "y", "colors"]), "`return_data` must be any combination of ['X','y','colors']"

    iris = load_iris()
    # Iris dataset
    X = pd.DataFrame(iris.data,
                     index = [*map(lambda x:f"iris_{x}", range(150))],
                     columns = [*map(lambda x: x.split(" (cm)")[0].replace(" ","_"), iris.feature_names)])

    y = pd.Series(iris.target,
                           index = X.index,
                           name = "Species")
    colors = map_colors(y, mode=1, palette=palette, desat=desat)#y.map(lambda x:{0:"red",1:"green",2:"blue"}[x])

    if noise is not None:
        X_noise = pd.DataFrame(
            np.random.RandomState(random_state).normal(size=(X.shape[0], noise)),
            index=X.index,
            columns=[*map(lambda j:f"noise_{j}", range(noise))]
        )
        X = pd.concat([X, X_noise], axis=1)
    if return_target_names:
        y = y.map(lambda k: iris.target_names[k])
    output = list()
    if "X" in return_data:
        output.append(X)
    if "y" in return_data:
        output.append(y)
    if "colors" in return_data:
        output.append(colors)
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)
