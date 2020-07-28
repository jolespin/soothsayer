
        
# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, warnings

# PyData
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean 

# Scikits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skbio.stats.ordination import pcoa, pcoa_biplot
from skbio import DistanceMatrix
from adjustText import adjust_text

from ..io import write_object
from ..utils import is_query_class, to_precision, dict_filter, is_symmetrical, is_nonstring_iterable, assert_acceptable_arguments
from ..symmetry import Symmetric
from ..visuals import plot_scatter

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

__all__ = ["MatrixDecomposition", "PrincipalComponentAnalysis","PrincipalCoordinatesAnalysis", "Manifold", "eigenprofiles_from_data"]
__all__ = sorted(__all__)


# General ordination methods
class CoreOrdinationMethods(object):
    # Get principal components
    def __getitem__(self, key):
        if type(key) == int:
            key = f"{prefix}{key}"
        return self.projection_[key]

        
    def _plot_biplot(
            self, 
            ax, 
            show_features,  
            precision, 
            use_percent, 
            method_top_features,
            arrowstyle,
            arrow_scale,
            color_arrow, 
            color_feature, 
            show_origin, 
            adjust_label_positions,
            autoscale_xpad, 
            autoscale_ypad, 
            arrow_kws, 
            arrow_properties,
            text_kws,
            **plot_scatter_kws,
            ):
            """
            Adapted from `erdogant/pca` on GitHub:
            https://github.com/erdogant/pca/blob/47d3dd101f6c757d783159f4f084f0dd38bf0412/pca/pca.py#L431
            to use skbio OrdinationResults object

            Create the Biplot based on model.
            
            Returns
            -------
            ax
            """
            warnings.warn("Using `show_features` to create a `biplot` is experimental.")
            
            # Pre-processing
            if show_features is True:
                show_features = -1
            if isinstance(show_features, int):
                assert show_features <= self.m, "`show_features` is larger than the number of features available.  Please use -1 to use all features."
                if show_features == -1:
                    # Set -1 to total number of features
                    show_features = self.m
                    
                # Get top features
                assert_acceptable_arguments(method_top_features, {"weighted", "partitioned", "importance"})
                number_of_requested_features = show_features
                if method_top_features == "weighted":
                    weighted_features = (self.contributions_.iloc[:,:2]*self.explained_variance_ratio_.iloc[:2]).sum(axis=1).sort_values(ascending=False)
                    show_features = weighted_features.iloc[:number_of_requested_features].index
                if method_top_features == "partitioned":
                    number_of_requested_features_is_even = (number_of_requested_features % 2) == 0
                    k = number_of_requested_features//2
                    # Even
                    if number_of_requested_features_is_even: 
                        pc1_features = self.contributions_.iloc[:,0].sort_values(ascending=False).iloc[:k].index.tolist()
                    # Odd
                    else:
                        pc1_features = self.contributions_.iloc[:,0].sort_values(ascending=False).iloc[:k+1].index.tolist()
                    pc2_features = self.contributions_.iloc[:,1].sort_values(ascending=False).iloc[:k].index.tolist()
                    show_features = sorted(set(pc1_features + pc2_features))
                if method_top_features == "importance":
                    # https://github.com/qiime2/q2-emperor/blob/756eddba1fc590f33d4bf7a3d0268517e68e9ae5/q2_emperor/_plot.py#L78
                    # select the top N most important features based on the vector's magnitude
                    show_features = self.importance_.iloc[:number_of_requested_features].index
                print("Selected {} features:".format(number_of_requested_features), list(show_features), sep="\n", file=sys.stderr)
                
            if isinstance(show_features, str):
                show_features = [show_features]
            if is_nonstring_iterable(show_features):
                assert set(show_features) <= set(self.loadings_.index), "`show_features` must be either an integer <= m (-1 to include all) or a feature subset of `data.columns`"
                
            # Get loading subset
            loadings = self.loadings_.loc[show_features,:].iloc[:,:2]

            # Pre-processing
            # Use the PCs only for scaling purposes
            x = self.projection_.iloc[:,0].values
            y = self.projection_.iloc[:,1].values

            mean_x = np.mean(x)
            max_x = np.abs(x).max()
            mean_y = np.mean(y)
            max_y = np.abs(y).max()

            # Keywords
            _arrow_kws = dict(color=color_arrow, linewidth=1, alpha=0.618) #, head_width=0.003, head_length = 0.005, alpha=0.382,  shape="left", overhang = 2.0, length_includes_head=False) #  head_width=0.1,
            _arrow_kws.update(arrow_kws)
            _arrowhead_kws = dict(arrowstyle=arrowstyle, mutation_scale=20, facecolor=color_arrow, alpha=0.618)
            
            _text_kws = dict(color=color_feature, ha='center', va='center')
            _text_kws.update(text_kws)

            # Plot arrows and text
            contributions = self.contributions_.loc[show_features,:].iloc[:,:2]
            text_objects = list()
            for i in range(0, loadings.shape[0]):
                feature = loadings.index[i]
                # Set PC1 vs PC2 direction. Note that these are not neccarily the best loading.
                xarrow = contributions.loc[feature].iloc[0] * np.sign(loadings.loc[feature].iloc[0]) * max_x * arrow_scale
                yarrow = contributions.loc[feature].iloc[1] * np.sign(loadings.loc[feature].iloc[1]) * max_y * arrow_scale

                # Plot arrow
                xyA = (mean_x, mean_y)
                xyB = (xarrow, yarrow)
                con = ConnectionPatch(xyA, xyB, coordsA="data", coordsB="data", **_arrowhead_kws)
                ax.plot([xyA[0], xyB[0]], [xyA[1], xyB[1]], **_arrow_kws)
                ax.add_artist(con)

                # Plot feature label
                text_objects.append(ax.text(xarrow*1.11, yarrow*1.11, feature,  **_text_kws))
                
            # Plot origin
            if show_origin:
                ax.scatter([mean_x], [mean_y], c=color_arrow, s=15)
                
            # Adjust label positions
            if adjust_label_positions:
                adjust_text(text_objects,  ax=ax)
                
            # Autoscale padding
            if autoscale_xpad is None:
                autoscale_xpad = 0
            if autoscale_ypad is None:
                autoscale_ypad = 0
                
            if any([autoscale_xpad, autoscale_ypad]):
                if bool(autoscale_xpad):
                    assert autoscale_xpad < 1
                    ax.set_xlim(*np.asarray(ax.get_xlim())*(1 + autoscale_xpad))
                if bool(autoscale_ypad):
                    assert autoscale_ypad < 1
                    ax.set_ylim(*np.asarray(ax.get_ylim())*(1 + autoscale_ypad))
                    
            return ax
        
    # Plotting
    def plot(
        self,    
        ax=None, 
        show_features=None,  
        precision=4, 
        use_percent=False, 
        method_top_features="partitioned",
        arrowstyle="-|>",
        arrow_scale=1,
        color_arrow="darkslategray", 
        color_feature="darkslategray",
        show_xaxis=True,
        show_yaxis=True,
        show_origin=True, 
        adjust_label_positions=False,
        autoscale_xpad=None, 
        autoscale_ypad=None, 
        arrow_kws=dict(), 
        arrow_properties=dict(),
        text_kws=dict(),
        axis_kws=dict(),
        **plot_scatter_kws,
        ):
        """
        See soothsayer.visuals.plot_scatter for details on parameters.
        """
        # Update scatter_kws
        plot_scatter_kws["ax"] = ax
        
        xlabel = f"{self.prefix}1"
        ylabel = f"{self.prefix}2"
        if hasattr(self, "explained_variance_ratio_"):
            if not use_percent:
                ev1 = to_precision(self.explained_variance_ratio_[f"{self.prefix}1"], precision=precision)
                ev2 = to_precision(self.explained_variance_ratio_[f"{self.prefix}2"], precision=precision)
                xlabel = f"{xlabel}({ev1})"
                ylabel = f"{ylabel}({ev2})"
            else:
                ev1 = to_precision(self.explained_variance_ratio_[f"{self.prefix}1"]*100, precision=precision)
                ev2 = to_precision(self.explained_variance_ratio_[f"{self.prefix}2"]*100, precision=precision)
                xlabel = f"{xlabel}({ev1}%)"
                ylabel = f"{ylabel}({ev2}%)" 
                
        # Plot scatter
        fig, ax, im = plot_scatter(data=self.projection_, x=f"{self.prefix}1", y=f"{self.prefix}2", xlabel=xlabel, ylabel=ylabel, **plot_scatter_kws)
        
        # Plot features (biplot)
        if not any([(show_features is False), (show_features is None)]):
            if hasattr(self, "loadings_"):
                assert self.loadings_ is not None, "Cannot visualize biplot for ordination objects without loadings.  If using `PrincipalCoordinatesAnalysis`, please rerun with `data` argument."
                # Plot biplot
                self._plot_biplot(
                    ax=ax,
                    show_features=show_features,
                    precision=precision,
                    use_percent=use_percent,
                    method_top_features=method_top_features,
                    arrowstyle=arrowstyle,
                    arrow_scale=arrow_scale,
                    color_arrow=color_arrow, 
                    color_feature=color_feature, 
                    show_origin=show_origin, 
                    adjust_label_positions=adjust_label_positions,
                    autoscale_xpad=autoscale_xpad, 
                    autoscale_ypad=autoscale_ypad, 
                    arrow_kws=arrow_kws, 
                    arrow_properties=arrow_properties,
                    text_kws=text_kws,
                )
            else:
                warnings.warn("Cannot visualize biplot for ordination objects without loadings")
                
        # Show axis
        _axis_kws = {"linestyle":"-", "color":"black", "linewidth":0.618, "alpha":0.1618}
        _axis_kws.update(axis_kws)
        if show_xaxis:
            ax.axhline(0, **_axis_kws)
        if show_yaxis:
            ax.axvline(0, **_axis_kws)
        return fig, ax, im
    

    # Plot Scree
    def plot_scree(
        self, 
        title=None, 
        target_explained_variance=None, 
        alpha_barchart=0.618, 
        alpha_marker=0.618, 
        alpha_target=0.618, 
        color_barchart="teal", 
        color_marker="black", 
        color_target="maroon",  
        show_xgrid=True, 
        show_ygrid=False, 
        fig_kws=dict(), 
        label_kws=dict(), 
        target_kws=dict(), 
        title_kws=dict(), 
        legend_kws=dict(),
        show_n_components=None, 
        ax=None, 
        bar_kws=dict(), 
        tick_kws=dict(), 
        line_kws=dict(), 
        style="seaborn-white",
        precision=4, 
        ):
        # Keywords
        _fig_kws = {"figsize":(8,3)}
        _fig_kws.update(fig_kws)
        _label_kws = {"fontsize":15}
        _label_kws.update(label_kws)
        _bar_kws = {"edgecolor":"black", "linewidth":1.618, "color":color_barchart, "alpha":alpha_barchart}
        _bar_kws.update(bar_kws)
        _line_kws = {"marker":"o",  "markeredgewidth":1, "markeredgecolor":"black", "color":color_marker, "alpha":alpha_marker}
        _line_kws.update(line_kws)
        _target_kws={"alpha":alpha_target, "color":color_target, "linestyle":":"}
        _target_kws.update(target_kws)
        _tick_kws = {"fontsize":12}
        _tick_kws.update(tick_kws)
        _title_kws = {"fontsize":15, "fontweight":"bold"}
        _title_kws.update(title_kws)
        _legend_kws = {"fontsize":12}
        _legend_kws.update(legend_kws)

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(**_fig_kws)
            else:
                fig = plt.gcf()
            # Plotting
            evr = self.explained_variance_ratio_
            evr[:show_n_components].plot(kind="bar",  ax=ax, label='_nolegend_', **_bar_kws)
            evr_cumsum = self.explained_variance_ratio_.cumsum()
            evr_cumsum[:show_n_components].plot(kind="line",  label='_nolegend_', ax=ax, **_line_kws)

            if target_explained_variance is not None:
                assert isinstance(target_explained_variance, float)
                idx_at_target = (evr_cumsum >= target_explained_variance).idxmax()
                if target_explained_variance:
                    x_pos = evr_cumsum.index.get_loc(idx_at_target)
                    y_pos = evr_cumsum[idx_at_target]
                    ax.plot([x_pos]*2, [0, y_pos], label="EV({}) ≥ {} [{}]".format(idx_at_target, to_precision(target_explained_variance, precision=precision), to_precision(y_pos, precision=precision)), **_target_kws)
                    ax.plot([min(ax.get_xlim()), x_pos], [y_pos, y_pos], **_target_kws)
                    ax.legend(**_legend_kws)


            # References
            ax.axhline(1.0, color="black", linestyle="-", linewidth=1)
            ax.set_ylabel("Explained Variance", **_label_kws)
            ax.set_xticklabels(ax.get_xticklabels(), **_tick_kws)

            if title is not None:
                ax.set_title(title, **_title_kws)
            ax.xaxis.grid(show_xgrid)
            ax.yaxis.grid(show_ygrid)

            ax.set_ylim((0,1.06180339887))

            return fig, ax
        
    def plot_feature_contribution(
        self,
        cmap=plt.cm.magma_r,
        vmin=0, 
        vmax=1,
        annot=True,
        linecolor="white",
        linewidth=1,
        title=None,
        show_eigenvectors=-1,
        show_features=-1,
        cbar_label="Explained variance ratio",
        ax=None,
        style="seaborn-white",
        fig_kws=dict(),
        heatmap_kws=dict(),
        title_kws=dict(fontsize=15, fontweight="bold")
        ):
        assert hasattr(self,"contributions_")
        assert self.contributions_ is not None
        if show_eigenvectors == -1:
            show_eigenvectors = self.contributions_.columns
        if show_features == -1:
            show_features = self.contributions_.index
        _heatmap_kws = dict(cmap=cmap, vmin=vmin, vmax=vmax, annot=annot, linecolor=linecolor, linewidth=linewidth, cbar_kws=dict(label=cbar_label))
        _heatmap_kws.update(heatmap_kws)
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(**fig_kws)
            else:
                fig = plt.gcf()
            sns.heatmap(self.contributions_.loc[show_features, show_eigenvectors], ax=ax, **_heatmap_kws)
            if title:
                ax.set_title(title, **title_kws)
        return ax
        

    # Save to file
    def to_file(self, path:str, compression="infer"):
        write_object(self, path, compression=compression)
        

# Sklearn Decomposition wrapper class
class MatrixDecomposition(object):
    """
    Wrapper for sklearn.decomposition objects
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
    """
    def __init__(self, data:pd.DataFrame, center=True, scale=False, n_components=-1, base_model=PCA, model_kws=dict(), prefix="PC.", name=None):
        self.data = data
#         self.data_transformed_ = data.copy()
        self.n, self.m = data.shape
        self.center = center
        self.scale = scale
        self.name = name
        self.prefix = prefix
        self.n_components = self._check_n_components(*data.shape, n_components)

        # Preprocess
        self.data_transformed_ = pd.DataFrame(
            data=StandardScaler(copy=True, with_mean=center, with_std=scale).fit_transform(data),
            index=data.index,
            columns=data.columns,
        )

        # Model
        self.model_ = self._build_model(base_model=base_model, data=self.data_transformed_, model_kws=model_kws)

        # Explained Variance
        if hasattr(self.model_, "explained_variance_ratio_"):
            self.explained_variance_ratio_ = pd.Series(
                self.model_.explained_variance_ratio_, 
                name="Explained variance ratio",
            )
            self.explained_variance_ratio_.index = self.explained_variance_ratio_.index.map(lambda i: f"{prefix}{i+1}")

        # Loadings
        if hasattr(self.model_, "components_"):
            self.loadings_ = pd.DataFrame(
                self.model_.components_.T, 
                index=self.data_transformed_.columns,
            )
            self.loadings_.columns = self.loadings_.columns.map(lambda i: f"{prefix}{i+1}")
            
        # Contributions
        if hasattr(self, "loadings_"):
            self.contributions_ = self.loadings_.abs()
            self.contributions_ = self.contributions_/self.contributions_.sum(axis=0)

        # Importance
        # https://github.com/qiime2/q2-emperor/blob/master/q2_emperor/_plot.py#L78
        if hasattr(self, "loadings_"):
            if self.loadings_ is not None:
                origin = np.zeros(self.loadings_.shape[1])
                self.importance_ = self.loadings_.apply(euclidean, axis=1, args=(origin,)).sort_values(ascending=False)
            else:
                self.importance_ = None
                
        # Project data onto eigenvectors
        self.projection_ =  pd.DataFrame(
            self.model_.transform(self.data_transformed_),
            index = self.data_transformed_.index,
        )
        self.projection_.columns = self.projection_.columns.map(lambda i: f"{prefix}{i+1}")

    # Check number of possible components
    def _check_n_components(self, n, m, n_components):
        possible_components = min(n, m)
        if n_components == -1:
            n_components = possible_components
        if (n_components > possible_components):
            n_components = possible_components
        return n_components

    # Construct model
    def _build_model(self, base_model, data, model_kws):
        model = base_model(n_components=self.n_components, **model_kws)
        model.fit(data)
        return model

# Principal Component Analysis
class PrincipalComponentAnalysis(CoreOrdinationMethods):
    """
    See soothsayer.ordination.MatrixDecomposition for details on parameters.
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    """
    def __init__(self, data:pd.DataFrame, center=True, scale=False, n_components=-1, base_model=PCA,  model_kws=dict(svd_solver="full", whiten=False), prefix="PC.", name=None):
        self.core_ = MatrixDecomposition(
            data=data,
            center=center,
            scale=scale,
            n_components=n_components,
            base_model=base_model,
            model_kws=model_kws,
            prefix=prefix,
            name=name,
        )
        for attr,obj in self.core_.__dict__.items():
            setattr(self, attr, obj)


# Principal Coordinant Analysis
class PrincipalCoordinatesAnalysis(CoreOrdinationMethods):
    """
    Wrapper for skbio.stats.ordination.pcoa
    http://scikit-bio.org/docs/latest/generated/generated/skbio.stats.ordination.pcoa.html#skbio.stats.ordination.pcoa
    """
    
    # Initialize
    def __init__(self, dism, data:pd.DataFrame=None,name=None, method='eigh', metric_type=None, node_type=None, prefix="PCoA."):
        if isinstance(dism, pd.DataFrame):
            dism = Symmetric(dism, name=name, edge_type=metric_type, association="dissimilarity")
        self.dism = dism
        if data is not None:
            assert set(dism.nodes) == set(data.index), "`dism` must have same sample set as `data.index`"
            data = data.loc[dism.nodes]
            self.n, self.m = data.shape
        self.data = data
        self.labels = dism.nodes
        self.node_type = node_type
        self.metric_type = metric_type
        self.name = name
        self.method = method
        self.prefix = prefix

        # Model
        self.model_ = self._build_model(dism=dism, data=data, base_model=pcoa)

        # Explained Variance
        if hasattr(self.model_, "proportion_explained"):
            self.explained_variance_ratio_ = pd.Series(
                self.model_.proportion_explained.values, 
                name="Explained variance ratio",
            )
            self.explained_variance_ratio_.index = self.explained_variance_ratio_.index.map(lambda i: f"{prefix}{i+1}")

        # Components
        if hasattr(self.model_, "eigvals"):
            self.eigvals_ = pd.Series(
                self.model_.eigvals.values, 
                name="Eigenvalues",
            )
            self.eigvals_.index = self.eigvals_.index.map(lambda i: f"{prefix}{i+1}")

        # Loadings
        if hasattr(self.model_, "features"):
            if self.model_.features is not None:
                self.loadings_ = self.model_.features.iloc[:,:self.m]
                self.loadings_.columns = self.loadings_.columns.map(lambda x: "{}{}".format(self.prefix, x[2:]))
            else:
                self.loadings_ = None
                
        # Contributions
        if hasattr(self, "loadings_"):
            if self.loadings_ is not None:
                self.contributions_ = self.loadings_.abs()
                self.contributions_ = self.contributions_/self.contributions_.sum(axis=0)
            else:
                self.contributions_ = None
                
        # Importance
        # https://github.com/qiime2/q2-emperor/blob/master/q2_emperor/_plot.py#L78
        if hasattr(self, "loadings_"):
            if self.loadings_ is not None:
                origin = np.zeros(self.loadings_.shape[1])
                self.importance_ = self.loadings_.apply(euclidean, axis=1, args=(origin,)).sort_values(ascending=False)
            else:
                self.importance_ = None
                
        # Project data onto eigenvectors
        self.projection_ =  pd.DataFrame(
            self.model_.samples.values,
            index = self.labels,
        )
        self.projection_.columns = self.projection_.columns.map(lambda i: f"{prefix}{i+1}")

    # Construct model
    def _build_model(self, dism, data, base_model):
        distance_matrix = DistanceMatrix(data=dism.values, ids=dism.nodes)
        ordination_results = base_model(distance_matrix)
        if data is not None:
            ordination_results = pcoa_biplot(ordination_results, data)
        return ordination_results

# Manifold Learning
class Manifold(CoreOrdinationMethods):
    """
    Wrapper for sklearn.manifold objects
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
    """
    def __init__(self, X:pd.DataFrame, base_model=TSNE, n_components=2, compute_pca="infer", n_components__pca=50, pca_kws=dict(center=True, scale=False), random_state=0, prefix="infer", name=None, **model_kws):
        # Init
        base_model_name = str(base_model)[:-2].split(".")[-1]
        self.name = name
        if prefix == "infer":
            if base_model_name == "TSNE":
                prefix = f"t-SNE."
            else:
                prefix = f"{base_model_name}."
        self.prefix = prefix

        # Is X precomputed?
        self.X = X.copy()
        self.X_input = self.X
        self.precomputed = self._check_precomputed(self.X, model_kws)
        if self.precomputed:
            compatible_for_precomputed = any(x == base_model_name for x in ["TSNE",  "MDS", "SpectralEmbedding", "spectral_embedding", "UMAP"])
            assert compatible_for_precomputed, f"{base_model_name} is not compatible for precomputed option"
            if base_model_name == "TSNE":
                model_kws["metric"] = "precomputed"
            if base_model_name == "MDS":
                model_kws["dissimilarity"] = "precomputed"
            if base_model_name == "SpectralEmbedding":
                model_kws["affinity"] = "precomputed"
            if base_model_name == "spectral_embedding":
                model_kws["adjacency"] = "precomputed"
        else:
            # Should PCA be computed?
            if compute_pca == "infer":
                if base_model_name == "TSNE":
                    compute_pca = True
            if compute_pca:
                # Check how many PCA components can be computed from the data
                self.n_components__pca = self._check_n_components(*self.X.shape, n_components=n_components__pca)

                self.X_input = PrincipalComponentAnalysis(data=self.X, base_model=PCA, n_components=self.n_components__pca, **pca_kws).projection_

        # Random state
        self.random_state = random_state
        compatible_for_rs = ["LocallyLinearEmbedding", "TSNE", "MDS", "SpectralEmbedding", "locally_linear_embedding", "smacof", "spectral_embedding", "UMAP"]
        if base_model_name in compatible_for_rs:
            model_kws["random_state"] = self.random_state

        # Check number of components
        self.n_components = self._check_n_components(*self.X_input.shape, n_components)


         # Model
        self.model = self._build_model(base_model=base_model, X_input=self.X_input, model_kws=model_kws)
        self.__dict__.update(dict_filter(self.model.__dict__, keys=lambda x:x.endswith("_")))

        # Get projections/embeddings
        if hasattr(self, "embedding_"):
            self.projection_ =  pd.DataFrame(
            self.embedding_,
            index = self.X_input.index,
        )
        self.projection_.columns = self.projection_.columns.map(lambda i: f"{prefix}{i+1}")


    # Check number of possible components
    def _check_n_components(self, n, m, n_components):

        possible_components = min(n, m)
        if n_components == -1:
            n_components = possible_components
        if (n_components > possible_components):
            warnings.warn(f"Data shape could not support {n_components} and is using the maximum {possible_components} instead.")
            n_components = possible_components
        return n_components

    # Check if data is precomputed
    def _check_precomputed(self, X, model_kws):
        # Preprocess
        precomputed = None
        if "metric" in model_kws:
            if model_kws["metric"] == "precomputed":
                precomputed = True
            else:
                precomputed = False
        if precomputed is None:
            if is_query_class(X, "Symmetric"):
                X = X.to_dense()
                precomputed = True
            else:
                precomputed = is_symmetrical(X, tol=1e-5)
        return precomputed

    # Construct model
    def _build_model(self, base_model, X_input, model_kws):
        model = base_model(n_components=self.n_components, **model_kws)
        model.fit(X_input)
        return model

    # Make compatible with scree plotting
    def plot_scree(self, **args):
        print("Cannot create scree plot for Manifolds", file=sys.stderr)
        return self

# =======================
# Wraps on wraps on wraps
# =======================
# Get eigenprofiles from data
def eigenprofiles_from_data(X:pd.DataFrame, y:pd.Series, class_type:str = None, mode="eigenprofiles", pca_kws=dict(),  axis=0, prefix="PC."):
    """
    Calculate eigenprofiles from data
    2018-July-13
    """
    accepted_modes = ["eigenprofiles", "eigenvalues", None]
    assert mode in accepted_modes, f"'{mode}' is not in accepted modes: {accepted_modes}"
    if axis == 1:
        X = X.T.copy()
    idx_attributes = X.columns
    profiles = list()
    d_eigenvalues = dict()

    if mode is None:
        tasks = ["eigenprofiles", "eigenvalues"]
    else:
        tasks = [mode]
    for group, _X in X.groupby(y, axis=0):
        _X_T = _X.T
        pca_model = PrincipalComponentAnalysis(_X_T, n_components=1, prefix=prefix, **pca_kws)
        if "eigenprofiles" in tasks:
            pc1 = pca_model.projection_[f"{prefix}1"]
            eigenprofile = pd.Series(pc1, index=idx_attributes, name=group)
            profiles.append(eigenprofile)
        if "eigenvalues" in tasks:
            eigenvalue = pca_model.explained_variance_ratio_[f"{prefix}1"]
            d_eigenvalues[group] = eigenvalue
    output = list()
    if "eigenprofiles" in tasks:
        df_eigenprofiles = pd.DataFrame(profiles)
        df_eigenprofiles.index.name = class_type
        output.append(df_eigenprofiles)
    if "eigenvalues" in tasks:
        eigenvalues = pd.Series(d_eigenvalues, name=class_type)
        output.append(eigenvalues)
    assert len(output) > 0, "Check `X`, `y`, and `mode` because no computation was done"
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)
