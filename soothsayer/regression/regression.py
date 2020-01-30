# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, warnings, re, time, copy
from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.metrics import r2_score
from scipy.stats import linregress

# Parallel
from joblib import Parallel, delayed

# Utility
from tqdm import tqdm

# Soothsayer
from ..utils import *
from ..io import write_object
from ..statistics import p_adjust
from ..visuals import draw_networkx_labels_with_box, plot_scatter

__all__ = ["MixedEffectsLinearModel", "GeneralizedLinearModel"]
__all__ = sorted(__all__)

# Core linear regression methods
class CoreLinearModel(object):

    # Check the regression variable format
    def check_regression_variable_format(self, label:str):
        alphanumeric = all(map(lambda x:x.isalnum() or (x == "_"), label))
        starts_with_alpha = label[0].isalpha()
        return all([alphanumeric, starts_with_alpha])

    # Relabel variables and attributes
    def relabel_attributes(self, df:pd.DataFrame, attribute_prefix="attr_"):
        n,m = df.shape
        df_relabeled = df.copy()
        idx_attrs_relabeled = [*map(lambda j:"{}{}".format(attribute_prefix, j), range(df.shape[1]))]
        encoding = dict(zip(df.columns, idx_attrs_relabeled))
        decoding = dict(zip(idx_attrs_relabeled, df.columns))
        df_relabeled.columns = idx_attrs_relabeled
        return df_relabeled, encoding, decoding

    # Decode the relabeled index
    def decode_index(self, indice):
        level_0, level_1 = indice
        id_attribute = re.split('\[T\..+]', level_0)[0]
        if id_attribute in self.decoding_:
            level_0 = level_0.replace(id_attribute, self.decoding_[id_attribute])
        return (level_0, level_1)

    # Convert a statsmodels table to pd.Series
    def statsmodels_table_to_pandas(self, model_results, name, multiple_comparison_method="fdr", residual_type="infer"):
        if is_query_class(model_results, "mixed"):
            residual_type = "resid"
        else:
            residual_type = "resid_response"
        assert hasattr(model_results, residual_type)
        statistics_data = OrderedDict([
            ("t_values",model_results.tvalues),
            ("p_values",model_results.pvalues),
            ("std_err", model_results.bse),
            ("coef", model_results.params),
            ("[0.025", model_results.conf_int().iloc[:,0]),
            ("0.975]", model_results.conf_int().iloc[:,1]),
            ])
        if multiple_comparison_method is not None:
            statistics_data["{}_values".format(multiple_comparison_method)] = p_adjust(model_results.pvalues, method=multiple_comparison_method)




        # Model
        model_data = {
            "scale":model_results.scale,
            "likelihood":model_results.llf,
            "converged":model_results.converged,
            "method":model_results.method,
            "num_observations":model_results.nobs,
            "r_squared":r2_score(model_results.y_actual, model_results.fittedvalues),
        }
        residuals = getattr(model_results, residual_type)
        model_data["mse"] = np.power(residuals, 2).mean()

        Se_statistics = pd.DataFrame(statistics_data).stack()
        Se_model = pd.Series(model_data)
        Se_model.index = pd.MultiIndex.from_tuples(Se_model.index.map(lambda x: ("Model", x)), names=[None,None])
        synopsis = pd.concat([Se_statistics, Se_model])
        synopsis.name = name
        return synopsis

    # Predict for a particular attribute
    def predict(self, attribute:str, Y:pd.DataFrame):
        assert self.fitted, "Please fit model first."
        Y = Y.copy()
        assert attribute in self.model_results_, "`{}` not in training data".format(attribute)
        # assert set(Y.columns) <= set(self.encoding_.keys()), "Not all variables in Y were used during fitting"
        Y.columns = Y.columns.map(lambda x: self.encoding_[x] if x in self.encoding_ else x)
        model = self.model_results_[attribute]
        return  model.predict(exog=Y)

    # Create directed networkx graph
    def create_graph(self, graph=None, test_field="fdr_values", tol_test=0.05, exclude_attributes:list=None, **attrs):
        assert self.fitted, "Please fit model before continuing."
        if tol_test is None:
            tol_test = 1.0
        assert 0 < tol_test <= 1.0, "0 < `tol_test` ≤ 1.0 "
        attrs.update({"test_field":test_field, "tol_test":tol_test})

        # Construct graph
        if graph is None:
            graph = nx.OrderedDiGraph(name=self.name, **attrs)
        assert isinstance(graph, (nx.DiGraph, nx.OrderedDiGraph)), "`graph` must either be DiGraph or OrderedDiGraph"
        self.graph = graph

        # Drop fields and attributes
        index = self.synopsis_.columns.get_level_values(0)
        drop_fields = list()
        for id_field in ["Intercept", "Group Var"]:
            if id_field in index:
                drop_fields.append(id_field)

        drop_attributes = list()
        if exclude_attributes is not None:
            drop_attributes += list(exclude_attributes)

        # Synopsis
        df_synopsis = self.synopsis_.copy().drop(drop_attributes, axis=0).drop(drop_fields, axis=1)
        assert test_field in df_synopsis.columns.get_level_values(1), "`{}` not available in `synopsis_`"

        # Add edges to graph
        for id_attribute, data in df_synopsis.iterrows():
            test_values = data[data.index.map(lambda x:x[1] == test_field)]
            mask_tol = test_values < tol_test
            if np.any(mask_tol):
                signficant_variables = test_values[mask_tol].index.get_level_values(0)
                coefficients = data[[*map(lambda x: (x, "coef"), signficant_variables)]]
                for (id_variable, _), coef in coefficients.iteritems():
                    self.graph.add_edge(id_variable, id_attribute, weight=np.abs(coef), sign=np.sign(coef), **data[id_variable].to_dict())
        # Check nodes
        number_of_nodes = self.graph.number_of_nodes()
        number_of_edges = self.graph.number_of_edges()

        if self.verbose:
            if number_of_nodes == 0:
                warnings.warn("Number of nodes is 0 with {} < {}".format(test_field, tol_test))
            else:
                print("N(Nodes) = {}, N(Edges) = {}".format(number_of_nodes, number_of_edges), file=sys.stderr)

        return self.graph

    # Plot graphical visualization
    def plot_graph(self,
                   pos=None,
                   show_node_labels=False,
                   node_colors=None,
                   edge_colors=None,
                   variable_color="#808080",
                   color_negative="#278198",
                   color_positive="#dc3a23",
                   node_sizes=500,
                   title=None,
                   title_kws=dict(),
                   func_edgewidth=1.618,
                   node_alpha=0.618,
                   edge_alpha=0.618,
                   node_edgewidths=1,
                   node_edgecolors="white",
                   font_color="black",
                   node_box_facecolor="darkgray",
                   node_box_edgecolor="white",
                   legend=None,
                   legend_markerscale=2,
                   legend_kws=dict(),
                   node_kws=dict(),
                   node_label_kws=dict(),
                   edge_kws = dict(),
                   figsize=(13,8),
                  ):
        assert hasattr(self, "graph"), "Please create graph before continuing."
        attributes = set(self.synopsis_.index) & set(self.graph.nodes())
        variables = set(self.graph.nodes()) - attributes
        # =====
        # Nodes
        # =====
        # Colors
        if node_colors is None:
            assert is_color(variable_color), "`variable_color = {}` is not a color".format(variable_color)
            node_colors = "#000000"
        if is_color(node_colors):
            node_colors = pd.concat([
                pd.Series([node_colors]*len(attributes), index=attributes),
                pd.Series([variable_color]*len(variables), index=variables),
            ])
        node_colors = pd.Series(node_colors)
        assert set(node_colors.index) >= set(attributes), "Not all attributes have node colors"
        for id_variable in variables:
            if id_variable not in node_colors:
                node_colors[id_variable] = variable_color
        node_colors = node_colors[list(self.graph.nodes())]

        # Sizes
        if is_number(node_sizes):
            node_sizes = pd.concat([
                pd.Series([node_sizes]*len(attributes), index=attributes),
                pd.Series([0]*len(variables), index=variables),
            ])
        node_sizes = pd.Series(node_sizes)
        assert set(node_sizes.index) >= set(attributes), "Not all attributes have node sizes"
        for id_variable in self.fixed_effect_variables_:
            node_sizes[id_variable] = 0
        node_sizes = node_sizes[list(self.graph.nodes())]

        # Labels
        if show_node_labels:
            if show_node_labels == True:
                show_node_labels = list(attributes)
            show_node_labels = dict(zip(show_node_labels, show_node_labels))
        # =====
        # Edges
        # =====
        # Weights
        coefficients = pd.Series(OrderedDict([*map(lambda edge_data: ((edge_data[0], edge_data[1]), edge_data[2]["coef"]), self.graph.edges(data=True))]))
        weights = coefficients.abs()
        if func_edgewidth is not None:
            if is_number(func_edgewidth):
                constant = func_edgewidth
                func_edgewidth = lambda w: float(constant)
            weights = weights.map(func_edgewidth)

        # Colors
        if edge_colors is None:
            edge_colors = coefficients.copy()
            edge_colors[coefficients > 0] = color_positive
            edge_colors[coefficients < 0] = color_negative
            edge_colors[coefficients == 0] = "black"

        # ========
        # Plotting
        # ========
        _title_kws = {"fontsize":15, "fontweight":"bold"}
        _title_kws.update(title_kws)
        _legend_kws = {'fontsize': 15, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
        _legend_kws.update(legend_kws)
        _node_kws = dict(node_color=node_colors.values, node_size=node_sizes.values, linewidths=node_edgewidths, edgecolors=node_edgecolors, alpha=node_alpha)
        _node_kws.update(node_kws)
        _node_label_kws = {"font_color":font_color, "font_size":12}
        _node_label_kws.update(node_label_kws)
        _edge_kws = dict(edge_color=edge_colors, width=weights, alpha=edge_alpha)
        _edge_kws.update(edge_kws)

        if pos is None:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="neato")
        self.pos = pos
        with plt.style.context("seaborn-white"):
            fig, ax = plt.subplots(figsize=figsize)
            nx.draw_networkx_edges(self.graph, pos=pos, ax=ax, **_edge_kws)
            nx.draw_networkx_nodes(self.graph, pos=pos,  ax=ax, **_node_kws)
            if show_node_labels:
                nx.draw_networkx_labels(self.graph, pos=pos, labels=show_node_labels, ax=ax, **_node_label_kws)
            draw_networkx_labels_with_box(self.graph, pos=pos, labels = variables, ax=ax, font_color=font_color, node_box_facecolor=node_box_facecolor, node_box_edgecolor=node_box_edgecolor)
            # Legend
            if legend is not None:
                ax.legend(*format_mpl_legend_handles(pd.Series(legend)),  markerscale=legend_markerscale, **_legend_kws)
            if title is not None:
                ax.set_title(title, **_title_kws)
            ax.axis("off")
            return fig, ax

    # Plot residuals
    def plot_residuals(self,
                       attribute:str,
                       show_attribute=True,
                       show_reference_fit=False,
                        c = None,
                        alpha=0.8,
                       title = None,
                       legend = None,
                       title_kws = dict(),
                       legend_kws = dict(),
                       scatter_kws=dict(),
                       annot_kws = dict(),

                       **kwargs,
                      ):
        assert self.fitted, "Please fit model first."
        assert attribute in self.model_results_, "`{}` not in training data".format(attribute)




        # Defaults
        _annot_kws = dict(xytext=(12, -12), va='top', xycoords='axes fraction', textcoords='offset points',fontsize=15)
        _annot_kws.update(annot_kws)
        _scatter_kws = dict(alpha=alpha, c=c)
        _scatter_kws.update(scatter_kws)
        _title_kws = {"fontsize":15, "fontweight":"bold"}
        _title_kws.update(title_kws)
        _legend_kws = {'fontsize': 15, 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
        _legend_kws.update(legend_kws)

        # Data
        data = pd.concat([self.model_results_[attribute].fittedvalues.to_frame("ŷ"), self.X_[attribute].to_frame("y")], axis=1)
        data["residuals"] = self.residuals_[attribute]
        r2 = self.synopsis_.loc[attribute,("Model", "r_squared")]

        # Plots
        with plt.style.context("seaborn-white"):
            fig, axes = plt.subplots(figsize=(13,5), ncols=2)
            # ------------
            # Scatter plot
            # ------------
            # Left
            ax = axes[0]

            if show_attribute:
                xlabel = "$ŷ$({})".format(attribute)
                ylabel = "$y$({})".format(attribute)
            else:
                xlabel = "$ŷ$"
                ylabel = "$y$"

            plot_scatter(data=data, x="ŷ", y="y",  ax=ax, xlabel=xlabel, ylabel=ylabel, **_scatter_kws)#, cmap=cmap, vmin=vmin,vmax=vmax)
            ax.annotate(s="$R^2$ = %0.5f"%(r2), xy=(0, 1),  **_annot_kws)

            if show_reference_fit:
                y_actual = data["y"]
                slope, intercept, r_value, p_value, std_err = linregress(y_actual,y_actual)
                ylim = ax.get_ylim()
                x_grid = np.linspace(*ylim)
                ax.plot(x_grid, x_grid*slope + intercept, color="black", linestyle=":")
                ax.set_ylim(ylim)

            # Right
            ax = axes[1]
            plot_scatter(data=data, x="ŷ", y="residuals",  ax=ax, xlabel=xlabel, ylabel="Residuals", **_scatter_kws)#, cmap=cmap, vmin=vmin,vmax=vmax)
            ax.axhline(0, color="black", linestyle="-", linewidth=1, zorder=0)

            # Legend
            if legend is not None:
                ax.legend(*format_mpl_legend_handles(legend), **_legend_kws)
            # Title
            if title is not None:
                fig.suptitle(title, **_title_kws)

            return fig, ax

    # Save object to file
    def to_file(self, path:str, **kwargs):
        path = format_path(path)
        write_object(self, path, **kwargs)

# Mixed Linear Effects Models
class MixedEffectsLinearModel(CoreLinearModel):
    """
    import soothsayer as sy

    # Testing Mixed Linear Model
    X_iris, y_iris = sy.utils.get_iris_data(return_data=["X","y"], noise=47)
    # Training Data
    fixed_effects = ["petal_width"]
    X = X_iris.drop(fixed_effects, axis=1)
    Y = pd.concat([y_iris.to_frame(), X_iris[fixed_effects]], axis=1)
    # Model
    model = sy.regression.MixedEffectsLinearModel()
    # Fit
    model.fit(X=X, Y=Y, fixed_effects_variables=fixed_effects, random_effect_variable="Species")
    # Create association graph
    model.create_graph(tol_test=0.05)
    # Save object (Important b/c statsmodels 0.10.0 is broken) #https://github.com/statsmodels/statsmodels/issues/5899
    model.to_file("iris.model.pbz2")
    # Load it (This part wouldn't work w/ the above comment)
    model = sy.io.read_object("iris.model.pbz2")
    # Test to make sure it looks good
    model.plot_graph(show_node_labels=True)
    # Show the data
    model.synopsis_.head()

    """
    def __init__(
            self,
            name=None,
            attr_type=None,
            obsv_type=None,
            multiple_comparison_method="fdr",
            n_jobs=1,
            verbose=True,
            mixedlm_kws=dict(),
            ):
        super().__init__()
        self.name = name
        self.attr_type = attr_type
        self.obsv_type = obsv_type
        self.multiple_comparison_method = multiple_comparison_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fitted = False
        self.mixedlm_kws = dict()

    # Fit model for each attribute
    def fit(self, X:pd.DataFrame, Y:pd.DataFrame, fixed_effects_variables:list, random_effect_variable:str, formula:str=None, add_attributes_to_fixed_variables=False, categorical_variables="infer", fit_kws=dict(), assert_missing_values=True):
        # Should probably bring these methods into CoreLinearModel
        start_time = time.time()
        # Check inputs
        if assert_missing_values:
            assert np.all(X.notnull()), "X has missing values and it will throw downstream errors."
            assert np.all(Y.notnull()), "Y has missing values and it will throw downstream errors."
        assert len(set(X.columns) & set(Y.columns)) == 0, "There can't be an overlap of columns in X and Y."
        assert set(X.index) == set(Y.index), f"X.index and Y.index must have same values"
        Y = Y.loc[X.index,:]
        self.X_ = X.copy()
        self.Y_ = Y.copy()
        self.formula = formula
        self.residual_type = "resid"

        # Adding attributes to fixed variables
        self.add_attributes_to_fixed_variables = add_attributes_to_fixed_variables
        if self.add_attributes_to_fixed_variables:
            warnings.warn("Experimental: `add_attributes_to_fixed_variables`")

        # Categorical variables
        if categorical_variables == "infer":
            categorical_variables = list()
            for id_variable in self.Y_.columns:
                if self.Y_[id_variable].dtype == object:
                    categorical_variables.append(id_variable)
            if self.verbose:
                if bool(categorical_variables):
                    print("Inferred categorical variables from `Y`: {}".format(categorical_variables), file=sys.stderr)

        if bool(categorical_variables):
            assert is_nonstring_iterable(categorical_variables), "`categorical_variables` must be a non-string iterable if it is not None"
            for id_variable in categorical_variables:
                assert id_variable in self.Y_.columns, "Not all variables from `categorical_variables` are in `Y.columns`"
                self.Y_[id_variable] = self.Y_[id_variable].astype("category")

        # Encode the variables
        X_relabeled, encoding_X, decoding_X = self.relabel_attributes(X, attribute_prefix="attr_")
        Y_relabeled, encoding_Y, decoding_Y = self.relabel_attributes(Y, attribute_prefix="meta_")
        self.encoding_ = {**encoding_X, **encoding_Y}
        self.decoding_ = {**decoding_X, **decoding_Y}

        # Check variables
        self.attributes_ = X.index
        self.fixed_effect_variables_ = [*map(lambda x:self.encoding_[x], fixed_effects_variables)]
        self.random_effect_variable_ = self.encoding_[random_effect_variable]
        variables = fixed_effects_variables + [random_effect_variable]
        assert set(variables) <= set(Y.columns), f"{set(variables) - set(Y.columns)} not in `Y.columns`"

        # Run regression models
        args = {
            "X":X_relabeled,
            "Y":Y_relabeled,
            "formula":formula,
            "multiple_comparison_method":self.multiple_comparison_method,
            "residual_type":self.residual_type,
            "fit_kws":fit_kws,
        }
        if self.verbose:
            data = Parallel(n_jobs=self.n_jobs)(delayed(self._run_mixedlm)(query_attr, **args) for query_attr in tqdm(X_relabeled.columns, "Modeling each attribute"))
        else:
            data = Parallel(n_jobs=self.n_jobs)(delayed(self._run_mixedlm)(query_attr, **args) for query_attr in X_relabeled.columns)

        # Model results
        self.models_ = OrderedDict(zip(X.columns, map(lambda x:x[0], data)))
        self.model_results_ = OrderedDict(zip(X.columns, map(lambda x:x[1], data)))
        self.synopsis_ = pd.DataFrame([*map(lambda x:x[2], data)])
        self.synopsis_.index.name = "id_attribute"
        self.residuals_ = pd.DataFrame(OrderedDict(zip(X.columns, map(lambda x:x[3], data))))

        # Decode the variables
        self.fixed_effect_variables_ = [*map(lambda x:self.decoding_[x], self.fixed_effect_variables_)]
        self.random_effect_variable_ = self.decoding_[self.random_effect_variable_]

        # Duration
        self.fitted = True
        self.duration_ = format_duration(start_time)
        return self

    # Run statsmodels
    def _run_mixedlm(self, query_attr, X, Y, formula, multiple_comparison_method, residual_type, fit_kws):

        # Independent variables
        fixed_effect_variables = copy.deepcopy(self.fixed_effect_variables_)
        variables = fixed_effect_variables + [self.random_effect_variable_]

        index = sorted(set(X[query_attr].dropna().index) & set(Y[variables].dropna(how="any", axis=0).index))
        X = X.loc[index]
        Y = Y.loc[index]
        design_matrix = Y.loc[:, variables]
        if self.add_attributes_to_fixed_variables:
            idx_attrs_fixed = list(set(X.columns) - set([query_attr]))
            design_matrix = pd.concat([design_matrix, X.loc[:,idx_attrs_fixed]], axis=1)
            fixed_effect_variables += idx_attrs_fixed

        # Add dependent variable
        y_actual = pd.Series(X[query_attr].astype(float), name=self.decoding_[query_attr])
        design_matrix["y"] = y_actual

        # Formula
        if formula is None: # Not sure how this will do with custom formula and 'add_attributes_to_fixed_variables'
            formula = "y ~ " + " + ".join(fixed_effect_variables)

        # Model
        model = smf.mixedlm(formula=formula, data=design_matrix, groups=design_matrix[self.random_effect_variable_], **self.mixedlm_kws)
        model_results = model.fit(**fit_kws)
        setattr(model_results, "y_actual", y_actual)

        # Parse results
        synopsis = self.statsmodels_table_to_pandas(model_results, name=self.decoding_[query_attr], multiple_comparison_method=multiple_comparison_method, residual_type=residual_type)
        synopsis.index = synopsis.index.map(self.decode_index)

        return (model, model_results, synopsis, getattr(model_results, residual_type))

# Generalized Linear Models
class GeneralizedLinearModel(CoreLinearModel):
    def __init__(
            self,
            family=None,
            name=None,
            attr_type=None,
            obsv_type=None,
            multiple_comparison_method="fdr",
            n_jobs=1,
            verbose=True,
            glm_kws=dict(),
            ):
        super().__init__()
        self.name = name
        self.attr_type = attr_type
        self.obsv_type = obsv_type
        self.multiple_comparison_method = multiple_comparison_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.fitted = False
        self.glm_kws = glm_kws
        if family is not None:
            self.glm_kws["family"] = family

    # Fit model for each attribute
    def fit(self, X:pd.DataFrame, Y:pd.DataFrame, fixed_effects_variables:list, formula:str=None, add_attributes_to_fixed_variables=False, categorical_variables="infer", residual_type="resid_response", fit_kws=dict(), assert_missing_values=True):
        start_time = time.time()
        # Check inputs
        if assert_missing_values:
            assert np.all(X.notnull()), "X has missing values and it will throw downstream errors."
            assert np.all(Y.notnull()), "Y has missing values and it will throw downstream errors."
        assert len(set(X.columns) & set(Y.columns)) == 0, "There can't be an overlap of columns in X and Y."
        assert set(X.index) == set(Y.index), f"X.index and Y.index must have same values"
        Y = Y.loc[X.index,:]
        self.X_ = X.copy()
        self.Y_ = Y.copy()
        self.formula = formula
        self.residual_type = residual_type

        # Adding attributes to fixed variables
        self.add_attributes_to_fixed_variables = add_attributes_to_fixed_variables
        if self.add_attributes_to_fixed_variables:
            warnings.warn("Experimental: `add_attributes_to_fixed_variables`")

        # Categorical variables
        if categorical_variables == "infer":
            categorical_variables = list()
            for id_variable in self.Y_.columns:
                if self.Y_[id_variable].dtype == object:
                    categorical_variables.append(id_variable)
            if self.verbose:
                if bool(categorical_variables):
                    print("Inferred categorical variables from `Y`: {}".format(categorical_variables), file=sys.stderr)

        if bool(categorical_variables):
            assert is_nonstring_iterable(categorical_variables), "`categorical_variables` must be a non-string iterable if it is not None"
            for id_variable in categorical_variables:
                assert id_variable in self.Y_.columns, "Not all variables from `categorical_variables` are in `Y.columns`"
                self.Y_[id_variable] = self.Y_[id_variable].astype("category")

        # Encode the variables
        X_relabeled, encoding_X, decoding_X = self.relabel_attributes(X, attribute_prefix="attr_")
        Y_relabeled, encoding_Y, decoding_Y = self.relabel_attributes(Y, attribute_prefix="meta_")
        self.encoding_ = {**encoding_X, **encoding_Y}
        self.decoding_ = {**decoding_X, **decoding_Y}

        # Check variables
        self.attributes_ = X.index
        self.fixed_effect_variables_ = [*map(lambda x:self.encoding_[x], fixed_effects_variables)]
        variables = fixed_effects_variables
        assert set(variables) <= set(Y.columns), f"{set(variables) - set(Y.columns)} not in `Y.columns`"

        # Run regression models
        args = {
            "X":X_relabeled,
            "Y":Y_relabeled,
            "formula":formula,
            "multiple_comparison_method":self.multiple_comparison_method,
            "residual_type":self.residual_type,
            "fit_kws":fit_kws,
        }
        if self.verbose:
            data = Parallel(n_jobs=self.n_jobs)(delayed(self._run_glm)(query_attr, **args) for query_attr in tqdm(X_relabeled.columns, "Modeling each attribute"))
        else:
            data = Parallel(n_jobs=self.n_jobs)(delayed(self._run_glm)(query_attr, **args) for query_attr in X_relabeled.columns)

        # Model results
        self.models_ = OrderedDict(zip(X.columns, map(lambda x:x[0], data)))
        self.model_results_ = OrderedDict(zip(X.columns, map(lambda x:x[1], data)))
        self.synopsis_ = pd.DataFrame([*map(lambda x:x[2], data)])
        self.synopsis_.index.name = "id_attribute"
        self.residuals_ = pd.DataFrame(OrderedDict(zip(X.columns, map(lambda x:x[3], data))))

        # Decode the variables
        self.fixed_effect_variables_ = [*map(lambda x:self.decoding_[x], self.fixed_effect_variables_)]

        # Duration
        self.fitted = True
        self.duration_ = format_duration(start_time)
        return self

    # Run statsmodels
    def _run_glm(self, query_attr, X, Y, formula, multiple_comparison_method, residual_type, fit_kws):

        # Independent variables
        fixed_effect_variables = copy.deepcopy(self.fixed_effect_variables_)
        variables = fixed_effect_variables

        # Handle missing values
        index = sorted(set(X[query_attr].dropna().index) & set(Y[variables].dropna(how="any", axis=0).index))
        X = X.loc[index]
        Y = Y.loc[index]

        design_matrix = Y.loc[:, variables]
        if self.add_attributes_to_fixed_variables:
            idx_attrs_fixed = list(set(X.columns) - set([query_attr]))
            design_matrix = pd.concat([design_matrix, X.loc[:,idx_attrs_fixed]], axis=1)
            fixed_effect_variables += idx_attrs_fixed


        # Add dependent variable
        y_actual = pd.Series(X[query_attr].astype(float), name=self.decoding_[query_attr])
        design_matrix["y"] = y_actual

        # Formula
        if formula is None: # Not sure how this will do with custom formula and 'add_attributes_to_fixed_variables'
            formula = "y ~ " + " + ".join(fixed_effect_variables)

        # Model
        model = smf.glm(formula=formula, data=design_matrix, **self.glm_kws)
        model_results = model.fit(**fit_kws)
        setattr(model_results, "y_actual", y_actual)
        # Parse results
        synopsis = self.statsmodels_table_to_pandas(model_results, name=self.decoding_[query_attr], multiple_comparison_method=multiple_comparison_method, residual_type=residual_type)
        synopsis.index = synopsis.index.map(self.decode_index)

        return (model, model_results, synopsis, getattr(model_results, residual_type))
