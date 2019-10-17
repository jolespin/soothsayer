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

__all__ = ["to_precision", "format_duration", "get_timestamp", "dataframe_to_matrixstring", "pad_left", "iterable_depth", "flatten", "get_unique_identifier", "infer_compression", "format_filename", "format_path","format_header", "boolean",
"dict_filter", "dict_reverse", "dict_expand", "dict_fill", "dict_build", "dict_collapse","dict_tree",
"rgb_to_rgba", "map_colors", "infer_cmap", "infer_vmin_vmax", "infer_continuous_type", "scalarmapping_from_data", "Chromatic", "create_logfile", "determine_mode_for_logfiles",
"is_dict", "is_rgb_like", "is_nonstring_iterable","is_dict_like", "is_color", "is_graph", "is_all_same_type", "is_number", "is_query_class","is_symmetrical", "is_in_namespace",
"format_mpl_legend_handles", "LEGEND_KWS", "DIVERGING_KWS", "CMAP_DIVERGING","COLOR_NEGATIVE", "COLOR_POSITIVE",  "get_coords_contour", "get_coords_centroid", "get_parameters_ellipse", "add_cbar_from_data", "configure_scatter",
"pd_series_collapse", "is_path_like", "pd_series_filter", "pd_dataframe_matmul", "pd_series_to_groupby_to_dataframe","pd_dataframe_query","contains","consecutive_replace", "force_symmetry","range_like","generate_random_sequence","fragment","pd_dataframe_extend_index","is_file_like","get_iris_data","assert_acceptable_arguments","filter_compositional","is_function","Command","get_directory_size","DisplayablePath","join_as_strings",
]
__all__ = sorted(__all__)


# =============
# Defaults
# =============
LEGEND_KWS = {'fontsize': 15, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
DIVERGING_KWS = dict(h_neg=220, h_pos=15, sep=20, s=90, l=50)
CMAP_DIVERGING = sns.diverging_palette(**DIVERGING_KWS, as_cmap=True)
COLOR_NEGATIVE, COLOR_POSITIVE = sns.diverging_palette(**DIVERGING_KWS, n=2).as_hex()

# ===========
# Assertions
# ===========
def assert_acceptable_arguments(query, target, operation="le", message=f"Invalid option provided.  Please refer to the following for acceptable arguments:"):
    """
    le: operator.le(a, b) : <=
    eq: operator.eq(a, b) : ==
    ge: operator.ge(a, b) : >=
    """
    if not is_nonstring_iterable(query):
        query = [query]
    query = set(query)
    target = set(target)
    func_operation = getattr(operator, operation)
    assert func_operation(query,target), "{}\n{}".format(message, target)
# ===========
# Types
# ===========
def is_function(obj):
    return hasattr(obj, "__call__")
def is_file_like(obj):
    return hasattr(obj, "read")
def is_dict(obj):
    return isinstance(obj, Mapping)
def is_rgb_like(c):
    condition_1 = type(c) != str
    condition_2 = len(c) in [3,4]
    return all([condition_1, condition_2])
def is_nonstring_iterable(obj):
    condition_1 = hasattr(obj, "__iter__")
    condition_2 =  not type(obj) == str
    return all([condition_1,condition_2])
def is_dict_like(obj):
    condition_1 = is_dict(obj)
    condition_2 = isinstance(obj, pd.Series)
    return any([condition_1, condition_2])
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
def is_all_same_type(iterable):
    iterable_types = set(map(lambda obj:type(obj), iterable))
    return len(iterable_types) == 1
def is_number(x, num_type = np.number):
    return np.issubdtype(type(x), num_type)
def is_query_class(x,query, case_sensitive=False):
    # Format single search queries
    if type(query) == str:
        query = [query]
    # Remove case if necessary
    x_classname = str(x.__class__)
    if not case_sensitive:
        x_classname = x_classname.lower()
        query = map(lambda q:q.lower(),query)
    # Check if any of the tags in query are in the input class
    verdict = any(q in x_classname for q in query)
    return verdict
def is_path_like(obj, path_must_exist=True):
    condition_1 = type(obj) == str
    condition_2 = hasattr(obj, "absolute")
    condition_3 = hasattr(obj, "path")
    obj_is_path_like = any([condition_1, condition_2, condition_3])
    if path_must_exist:
        if obj_is_path_like:
            return os.path.exists(obj)
        else:
            return False
    else:
        return obj_is_path_like
# def is_symmetrical(X:pd.DataFrame, tol=None):
#     if X.shape[0] != X.shape[1]:
#         return False
#     if X.shape[0] == X.shape[1]:
#         if tol is None:
#             return np.all(np.tril(X) == np.triu(X).T)
#         if tol:
#             return (np.tril(X) - np.triu(X).T).ravel().min() < tol
def is_in_namespace(variable_names, namespace, func_logic=all):
    """
    Check if variable names are in the namespace (i.e. globals())
    """
    assert hasattr(variable_names, "__iter__"), f"`variable_names` should be either a single string on an object or an iterable of strings of variable names"
    if type(variable_names) == str:
        variable_names = [variable_names]
    namespace = set(namespace)
    return func_logic(map(lambda x: x in namespace, variable_names))
# Boolean
def boolean(x, true_values={"true", "t", "yes", "1"}, false_values={"false", "f", "no", "0"}, assertion_message="Please choose either: 'True' or 'False'"):
    """
    Not case sensitive
    """
    x = str(x).lower()
    option = None
    if x in list(map(str,true_values)):
        option = True
    if x in list(map(str,false_values)):
        option = False
    assert option is not None, assertion_message
    return option


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
# Miscellaneous
# =============
# Random sequence generation
def generate_random_sequence(size:int=100, alphabet=["A","T", "C", "G"], weights=[0.25,0.25,0.25,0.25], random_state=None):
    rs = np.random.RandomState(random_state)
    x = np.random.choice(alphabet, size=size, replace=True, p=weights)
    return "".join(x)
# Truncate a float by a certain precision
def to_precision(x, precision=5, into=float):
    return into(("{0:.%ie}" % (precision-1)).format(x))
# Get duration
def format_duration(start_time):
    """
    Adapted from @john-fouhy:
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
# Get current timestamp
def get_timestamp(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.utcnow().strftime(fmt)
# For ete3 ClusterNode
def dataframe_to_matrixstring(df):
    return df.to_csv(None, sep="\t",index_label="#Names")
# Left padding
def pad_left(x, block_size=3, fill=0):
    """
    Pad a string representation of digits
    """
    if len(x) > block_size:
        return x
    else:
        right = np.array(list(str(x)))
        left = np.repeat(str(fill), block_size - right.size )
        return "".join(np.concatenate([left, right]))
# Join as strings
def join_as_strings(*args, delimiter:str="_" ):
    return delimiter.join(list(map(str, args)))
# =============
# Iterables
# =============
# Fragment a sequence string
def fragment(seq:str, K:int=5, step:int=1, overlap:bool=False):
    K = int(K)
    step = int(step)
    if not overlap:
        step = K
    iterable = range(0, len(seq) - K + 1, step)
    for i in iterable:
        frag = seq[i:i+K]
        yield frag

# Get depth of an iterable
def iterable_depth(arg, exclude=None):
    # Adapted from the following SO post:
    # https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
    # @marco-sulla
    exclude = set([str])
    if exclude is not None:
        if not hasattr(exclude, "__iter__"):
            exclude = [exclude]
        exclude.update(exclude)

    if isinstance(arg, tuple(exclude)):
        return 0

    try:
        if next(iter(arg)) is arg:  # avoid infinite loops
            return 1
    except TypeError:
        return 0

    try:
        depths_in = map(lambda x: iterable_depth(x, exclude), arg.values())
    except AttributeError:
        try:
            depths_in = map(lambda x: iterable_depth(x, exclude), arg)
        except TypeError:
            return 0
    try:
        depth_in = max(depths_in)
    except ValueError:
        depth_in = 0

    return 1 + depth_in

# Flatten nested iterables
def flatten(nested_iterable, into=list, **args_iterable):
    # Adapted from @wim:
    # https://stackoverflow.com/questions/16312257/flatten-an-iterable-of-iterables
    def _func_recursive(nested_iterable):
        for x in nested_iterable:
            if hasattr(x, "__iter__") and not isinstance(x, str):
                yield from flatten(x)
            else:
                yield x
    # Unpack data
    data_flattened = [*_func_recursive(nested_iterable)]
    # Convert type
    return into(data_flattened, **args_iterable)

# Range like input data
def range_like(data, start=0):
    return np.arange(len(data)) + start

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
# Format file path
def format_path(path, into=str):
    assert not is_file_like(path), "`path` cannot be file-like"
    if hasattr(path, "absolute"):
        path = str(path.absolute())
    if hasattr(path, "path"):
        path = str(path.path)
    return into(path)

# Format header for printing
def format_header(text:str, line_character="=", n=None):
    if n is None:
        n = len(text)
    line = n*line_character
    return "{}\n{}\n{}".format(line, text, line)
# Consecutive replace on a string
def consecutive_replace(x:str, *patterns):
    if len(patterns) == 1:
        patterns = patterns[0]
    for (a,b) in patterns:
        x = x.replace(a,b)
    return x

# ============
# Dictionaries
# ============
# Dictionary as a tree
def dict_tree():
    """
    Source: https://gist.github.com/hrldcpr/2012250
    """
    return defaultdict(dict_tree)

# Reverse a dictionary
def dict_reverse(d):
    into = type(d)
    data = [(v,k) for k,v in d.items()]
    return into(data)

# Expand dictionary
def dict_expand(d, into=pd.Series, *args):
    """
    Convert {group:[elements]} ==> {element[i]:group[j]}
    """
    return into(OrderedDict((r,p) for p,q in d.items() for r in q), *args)

# Fill dictionary
def dict_fill(d, index, filler_value="#FFFFF0", into=dict):
    data = [(k,filler_value) for k in index if k not in d] + list(d.items()) #"#8e8f99"
    return into(data)

# Build a dictionary from repeated elements
def dict_build(input_data, into=dict):
    """
    input_data: [(value, iterable)]
    d_output: {key_in_iterable:value}
    """
    d_output = OrderedDict()
    for value, iterable in input_data:
        for key in iterable:
            d_output[key] = value
    return into(d_output)

# Fold dictionary
def dict_collapse(d, into=dict):
    """
    Folds dictionary into dict of lists
    """
    d_collapsed = defaultdict(list)
    for k,v in d.items():
        d_collapsed[v].append(k)
    return into(d_collapsed)

# Subset a dictionary
def dict_filter(d, keys, into=dict):
    """
    keys can be an iterable or function
    """
    if hasattr(keys, "__call__"):
        f = keys
        keys = filter(f, d.keys())
    return into(map(lambda k:(k,d[k]), keys))


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
    idx_query = pd.DataFrame(masks).sum(axis=0).compress(lambda x: x >= tol).index
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

# =======
# Filters
# =======
def contains(query, include, exclude=None):
    """
    Is anything from `include` in `query` that doesn't include anything from `exclude`
    `query` can be any iterator that is not a generator
    """
    if type(include) == str:
        include = [include]
    condition_A = any(x in query for x in include)
    if exclude is not None:
        if type(exclude) == str:
            exclude = [exclude]
        condition_B = all(x not in query for x in exclude)
        return all([condition_A, condition_B])
    else:
        return condition_A

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
        return data.compress(lambda x: operation(x,tol)).index

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
# ===============
# Shell utilities
# ===============
# View directory structures
class DisplayablePath(object):
    """
    Display the tree structure of a directory.

    Implementation adapted from the following sources:
        * Credits to @abstrus
        https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = pathlib.Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = pathlib.Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))

    # Additions by Josh L. Espinoza for Soothsayer
    @classmethod
    def get_ascii(cls, root):
        ascii_output = list()
        paths = cls.make_tree(root)
        for path in paths:
            ascii_output.append(path.displayable())
        return "\n".join(ascii_output)
    @classmethod
    def view_directory_tree(cls, root, file=sys.stdout):
        print(cls.get_ascii(root), file=file)

# Directory size
def get_directory_size(path_directory='.'):
    """
    Adapted from @Chris:
    https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    """
    path_directory = format_path(path_directory)

    total_size = 0
    seen = {}
    for dirpath, dirnames, filenames in os.walk(path_directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                stat = os.stat(fp)
            except OSError:
                continue

            try:
                seen[stat.st_ino]
            except KeyError:
                seen[stat.st_ino] = True
            else:
                continue

            total_size += stat.st_size

    return total_size

# Bash commands
class Command(object):
    """
    Run bash commands and stuff.

    Recommended usage:
    ------------------
    with open("test_commands.sh", "w") as f_cmds:
        cmd = Command("echo ':)' > testing_output.txt", name="TEST", f_cmds=f_cmds)
        cmd.run(epilogue="footer", prologue="header", checkpoint="testing_output.txt.checkpoint")

    or

    f_cmds = open("test_commands.sh", "w")
    cmd = Command("echo ':)'' > testing_output.txt", name="TEST", f_cmds=f_cmds)
    cmd.run(epilogue="footer", prologue="header", checkpoint="testing_output.txt.checkpoint")
    f_cmds.close()

    Just in case you need a quick one-liner [not recommended but works]:
    -------------------------------------------------------------------
    cmd = Command("echo ':)'' > testing_output.txt", name="TEST", f_cmds="test_commands.sh")
    cmd.run(epilogue="footer", prologue="header", checkpoint="testing_output.txt.checkpoint").close()

    or

    cmd = Command("echo ':)'' > testing_output.txt", name="TEST", f_cmds="test_commands.sh")
    cmd.run(epilogue="footer", prologue="header", checkpoint="testing_output.txt.checkpoint")
    cmd.close()

    Future:
    -------
    * Create an object called ExecutablePipeline that wraps Command objects together
    * Something like this:
        ep = ExecutablePipeline(name="RNA-seq mapping", description="Quality trim, remove contaminants, and map reads to reference")
        # This method
        ep.create_step(name="kneaddata", pos=1, checkpoint="path/to/checkpoint", write_stdout="path/to/stdout", write_stderr="path/to/stderr", write_returncode="path/to/returncode")
        ep["kneaddata"].set_inputs(*?)
        ep["kneaddata"].set_outputs(*?)
        # or this method
        ep.create_step(name="kneaddata", pos=1, checkpoint="path/to/checkpoint", write_stdout="path/to/stdout", write_stderr="path/to/stderr", write_returncode="path/to/returncode", inputs=*?, outputs=*?)
        ep.execute(?*)

    Here is an example for constructing pipelines:
    -------------------------------------------------------------------
    # =========
    # Utility
    # =========
    def process_command(cmd, f_cmds, logfile_name, description, directories, io_filepaths):
        start_time = time.time()
        # Info
        program = logfile_name.split("_")[-1]
        print(description, file=sys.stdout)
        print("Input: ", io_filepaths[0], "\n", "Output: ", io_filepaths[1], sep="", file=sys.stdout)
        print("Command: ", " ".join(cmd), file=sys.stdout)
        executable = Command(cmd, name=logfile_name, description=description, f_cmds=f_cmds)
        executable.run(
            prologue=format_header(program, "_"),
            dry="infer",
            errors_ok=False,
            error_message="Check the following files: {}".format(os.path.join(directories["log"], "{}.*".format(logfile_name))),
            checkpoint=os.path.join(directories["checkpoints"], "{}".format(logfile_name)),
            write_stdout=os.path.join(directories["log"], "{}.o".format(logfile_name)),
            write_stderr=os.path.join(directories["log"], "{}.e".format(logfile_name)),
            write_returncode=os.path.join(directories["log"], "{}.returncode".format(logfile_name)),
            f_verbose=sys.stdout,
        )
        print("Duration: {}".format(executable.duration_), file=sys.stdout)
        return executable
    # =========
    # Kneaddata
    # =========
    program = "kneaddata"

    # Add to directories
    output_directory = directories[("intermediate", program)] = create_directory(os.path.join(directories["intermediate"], "{}_output".format(program)))

    # Info
    step = "1"
    logfile_name = "{}_{}".format(step, program)
    description = "{}. {} | Removing human associated reads and quality trimming".format(step, program)

    # i/o
    input_filepath = [opts.r1, opts.r2]
    output_filename = ["kneaddata_repaired_1.fastq.gz", "kneaddata_repaired_2.fastq.gz"]
    output_filepath = list(map(lambda filename: os.path.join(output_directory, filename), output_filename))
    io_filepaths = [input_filepath,  output_filepath]

    # Parameters
    params = {
        "reads_r1":input_filepath[0],
        "reads_r2":input_filepath[1],
        "output_directory":output_directory,
        "opts":opts,
        "directories":directories,
    }
    cmd = get_kneaddata_cmd(**params)
    process = process_command(cmd, f_cmds=f_cmds, logfile_name=logfile_name, description=description, directories=directories, io_filepaths=io_filepaths)
    sys.stdout.flush()

    """
    def __init__(self, *args, name=None, description=None, f_cmds=sys.stdout):
        if len(args) == 1:
            args = args[0]
            if isinstance(args, str):
                args = [args]
        cmd = " ".join(args)
        self.cmd = cmd
        if not is_file_like(f_cmds):
            f_cmds = open(f_cmds, "w")
        self.f_cmds = f_cmds
        self.name = name
        self.description = description

    def __repr__(self):
        class_name = str(self.__class__)[17:-2]
        return '{}(name={}, description={}, cmd="{}")'.format(class_name, self.name, self.description, self.cmd)
    def close(self):
        self.f_cmds.close()
        return self
    def _write_output(self, data, filepath):
        if filepath is not None:
            if not is_file_like(filepath):
                filepath = format_path(filepath)
                f_out = open(filepath, "w")
            else:
                f_out = filepath
            print(data, file=f_out)
            if f_out not in {sys.stdout, sys.stderr}:
                f_out.close()

    # Run command
    def run(self, prologue=None, epilogue=None, errors_ok=False, dry="infer", checkpoint=None, write_stdout=None, write_stderr=None, write_returncode=None, close_file=False, checkpoint_message_notexists="Running...", checkpoint_message_exists="Loading...", error_message=None, f_verbose=sys.stdout):
        """
        Should future versions should have separate prologue and epilogue for f_cmds and f_verbose?
        """
        # ----------
        # Checkpoint
        # ----------
        if checkpoint is not None:
            checkpoint = format_path(checkpoint)
        if dry == "infer":
            dry = False
            if checkpoint is not None:
                if os.path.exists(checkpoint):
                    dry = True
                    if checkpoint_message_exists is not None:
                        print(checkpoint_message_exists, file=f_verbose)
                        f_verbose.flush()
                else:
                    if checkpoint_message_notexists is not None:
                        print(checkpoint_message_notexists, file=f_verbose)
                        f_verbose.flush()

        # ----
        # Info
        # ----
        if self.f_cmds is not None:
            # Prologue
            if prologue is not None:
                self.prologue_ = prologue
                print("#", prologue, file=self.f_cmds)
            # Command
            print(self.cmd, file=self.f_cmds)
            # Epilogue
            if epilogue is not None:
                self.epilogue_ = epilogue
                print("#", epilogue, file=self.f_cmds)
            self.f_cmds.flush()
            if self.f_cmds not in {sys.stdout, sys.stderr}:
                os.fsync(self.f_cmds.fileno())
        # Run
        if not dry:
            start_time = time.time()
            self.process_ = subprocess.Popen(self.cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process_.wait()
            self.returncode_ = self.process_.returncode
            self.duration_ = format_duration(start_time)

            # stdout
            self.stdout_ = self.process_.stdout.read()
            if isinstance(self.stdout_, bytes):
                self.stdout_ = self.stdout_.decode("utf-8")
            self._write_output(data=self.stdout_, filepath=write_stdout)
            # stderr
            self.stderr_ = self.process_.stderr.read()
            if isinstance(self.stderr_, bytes):
                self.stderr_ = self.stderr_.decode("utf-8")
            self._write_output(data=self.stderr_, filepath=write_stderr)
            # Return code
            self._write_output(data=self.returncode_, filepath=write_returncode)

            # Check
            if not errors_ok:
                if self.returncode_ != 0:
                    # print('\nThe following command failed with returncode = {}:\n"{}"'.format(self.returncode_, self.cmd), file=f_verbose)
                    if error_message is not None:
                        print(error_message, file=f_verbose)
                    sys.exit(self.returncode_)

            # Create checkpoint
            if checkpoint is not None:
                if self.returncode_ == 0:
                    duration = format_duration(start_time)
                    with open(checkpoint, "w") as f_checkpoint:
                        print(get_timestamp(), duration, file=f_checkpoint)
        # Close file object
        if self.f_cmds not in {None, sys.stdout, sys.stderr}:
            if close_file:
                self.close()
        return self
