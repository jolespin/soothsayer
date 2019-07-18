# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time

# PyData
import pandas as pd

# Scikits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skbio.stats.ordination import pcoa
from ..io import write_object
from ..utils import is_query_class, to_precision, dict_filter, is_symmetrical
from ..symmetry import Symmetric
from ..visuals import plot_scatter

__all__ = ["MatrixDecomposition", "PrincipalComponentAnalysis","PrincipalCoordinatesAnalysis", "Manifold", "eigenprofiles_from_data"]
__all__ = sorted(__all__)


# General ordination methods
class CoreOrdinationMethods(object):
    # Get principal components
    def __getitem__(self, key):
        if type(key) == int:
            key = f"{prefix}{key}"
        return self.projection_[key]

    # Plotting
    def plot(self, precision=4, **args):
        """
        See soothsayer.visuals.plot_scatter for details on parameters.
        """
        xlabel = f"{self.prefix}1"
        ylabel = f"{self.prefix}2"
        if hasattr(self, "explained_variance_ratio_"):
            ev1 = to_precision(self.explained_variance_ratio_[f"{self.prefix}1"], precision=precision)
            ev2 = to_precision(self.explained_variance_ratio_[f"{self.prefix}2"], precision=precision)
            xlabel = f"{xlabel}({ev1})"
            ylabel = f"{ylabel}({ev2})"
        return plot_scatter(data=self.projection_, x=f"{self.prefix}1", y=f"{self.prefix}2", xlabel=xlabel, ylabel=ylabel, **args)

    # Plot Scree
    def plot_scree(self, title=None, color="teal", fig_kws=dict(), label_kws=dict(), title_kws=dict(), truncate=None, ax=None, mode="explained_variance", bar_kws=dict(), tick_kws=dict(), line_kws=dict(), style="seaborn-white"):
        _fig_kws = {"figsize":(8,3)}
        _fig_kws.update(fig_kws)
        _label_kws = {"fontsize":15}
        _label_kws.update(label_kws)
        _bar_kws = {"edgecolor":"black", "linewidth":1}
        _bar_kws.update(bar_kws)
        _line_kws = {"marker":"o",  "markeredgewidth":1, "markeredgecolor":"black"}
        _line_kws.update(line_kws)
        _tick_kws = {"fontsize":12}
        _tick_kws.update(tick_kws)
        _title_kws = {"fontsize":15, "fontweight":"bold"}
        _title_kws.update(title_kws)

        with plt.style.context(style):
            if ax is not None:
                fig = plt.gcf()
            else:
                fig, ax = plt.subplots(**_fig_kws)

            if mode == "explained_variance":
                self.explained_variance_ratio_[:truncate].plot(kind="bar",  color=color, ax=ax, **_bar_kws)
                ax.set_ylabel("Explained Variance", **_label_kws)
            if mode == "cumulative_variance":
                self.explained_variance_ratio_.cumsum()[:truncate].plot(kind="line", color=color, ax=ax, **_line_kws)
                ax.set_ylabel("Cumulative Explained Variance", **_label_kws)
            ax.set_xticklabels(ax.get_xticklabels(), **_tick_kws)
            if title is not None:
                ax.set_title(title, **_title_kws)
            return fig, ax

    # Save to file
    def to_file(self, path:str, compression="infer"):
        write_object(self, path, compression=compression)

# Sklearn Decomposition wrapper class
class MatrixDecomposition(object):
    """
    Wrapper for sklearn.decomposition objects
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
    """
    def __init__(self, X:pd.DataFrame, center=True, scale=False, n_components=-1, base_model=PCA, model_kws=dict(), prefix="PC.", name=None, verbose=False):
        self.X = X.copy()
        self.X_input = self.X
        self.center = center
        self.scale = scale
        self.name = name
        self.prefix = prefix
        self.verbose = verbose
        self.n_components = self._check_n_components(*X.shape, n_components)

        # Preprocess
        if self.center == True:
            if verbose:
                print("Centering data", file=sys.stderr)
            self.X_input = self.X_input - self.X_input.mean(axis=0)
        if self.scale == True:
            if verbose:
                print("Scaling data", file=sys.stderr)
            self.X_input = self.X_input/self.X_input.std(axis=0)

        # Model
        self.model = self._build_model(base_model=base_model, X_input=self.X_input, model_kws=model_kws)

        # Explained Variance
        if hasattr(self.model, "explained_variance_ratio_"):
            if verbose:
                print("Adding`explained_variance_ratio_`", file=sys.stderr)
            self.explained_variance_ratio_ = pd.Series(self.model.explained_variance_ratio_, name="Explained variance ratio")
            self.explained_variance_ratio_.index = self.explained_variance_ratio_.index.map(lambda i: f"{prefix}{i+1}")

        # Components
        if hasattr(self.model, "components_"):
            if verbose:
                print("Adding`components_`", file=sys.stderr)
            self.components_ = self.model.components_

        # Project data onto eigenvectors
        if verbose:
            print("Adding`projection_`", file=sys.stderr)
        self.projection_ =  pd.DataFrame(
            self.model.transform(self.X_input),
            index = self.X_input.index,
        )
        self.projection_.columns = self.projection_.columns.map(lambda i: f"{prefix}{i+1}")

    # Check number of possible components
    def _check_n_components(self, n, m, n_components):
        possible_components = min(n, m)
        if n_components == -1:
            n_components = possible_components
        if (n_components > possible_components):
            if self.verbose:
                print(f"Data shape could not support {n_components} and is using the maximum {possible_components} instead.", file=sys.stderr)
            n_components = possible_components
        return n_components

    # Construct model
    def _build_model(self, base_model, X_input, model_kws):
        model = base_model(n_components=self.n_components, **model_kws)
        model.fit(X_input)
        return model

# Principal Component Analysis
class PrincipalComponentAnalysis(CoreOrdinationMethods):
    """
    See soothsayer.ordination.MatrixDecomposition for details on parameters.
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    """
    def __init__(self, X:pd.DataFrame, center=True, scale=False, n_components=-1, base_model=PCA, model_kws=dict(), prefix="PC.", name=None, verbose=False):

        self.core = MatrixDecomposition(
            X=X,
            center=center,
            scale=scale,
            n_components=n_components,
            base_model=base_model,
            model_kws=model_kws,
            prefix=prefix,
            name=name,
            verbose=verbose
        )
        self.__dict__.update(self.core.__dict__)


# Principal Coordinant Analysis
class PrincipalCoordinatesAnalysis(CoreOrdinationMethods):
    """
    Wrapper for skbio.stats.ordination.pcoa
    http://scikit-bio.org/docs/latest/generated/generated/skbio.stats.ordination.pcoa.html#skbio.stats.ordination.pcoa
    """
    def __init__(self, kernel, metric_type=None, node_type=None, prefix="PCoA.", name=None, verbose=False, symmetric_kws=dict()):
        if is_query_class(kernel, "DataFrame"):
            kernel = Symmetric(kernel, name=name, metric_type=metric_type, mode="dissimilarity", **symmetric_kws)
        self.kernel = kernel
        self.labels = kernel.labels
        self.node_type = node_type
        self.metric_type = metric_type
        self.name = name
        self.prefix = prefix
        self.verbose = verbose

        # Model
        self.model = self._build_model(kernel=kernel, base_model=pcoa)

        # Explained Variance
        if hasattr(self.model, "proportion_explained"):
            if verbose:
                print("Adding`explained_variance_ratio_`", file=sys.stderr)
            self.explained_variance_ratio_ = pd.Series(self.model.proportion_explained.values, name="Explained variance ratio")
            self.explained_variance_ratio_.index = self.explained_variance_ratio_.index.map(lambda i: f"{prefix}{i+1}")

        # Components
        if hasattr(self.model, "eigvals"):
            if verbose:
                print("Adding`eigvals_`", file=sys.stderr)
            self.eigvals_ = pd.Series(self.model.eigvals.values, name="Eigenvalues")
            self.eigvals_.index = self.eigvals_.index.map(lambda i: f"{prefix}{i+1}")


        # Project data onto eigenvectors
        if verbose:
            print("Adding`projection_`", file=sys.stderr)
        self.projection_ =  pd.DataFrame(
            self.model.samples.values,
            index = self.labels,
        )
        self.projection_.columns = self.projection_.columns.map(lambda i: f"{prefix}{i+1}")

    # Construct model
    def _build_model(self, kernel, base_model):
        df_dism = kernel.to_dense()
        model = base_model(df_dism)
        return model

# Manifold Learning
class Manifold(CoreOrdinationMethods):
    """
    Wrapper for sklearn.manifold objects
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
    """
    def __init__(self, X:pd.DataFrame, base_model=TSNE, n_components=2, compute_pca="infer", n_components__pca=50, pca_kws=dict(center=True, scale=False), random_state=0, prefix="infer", name=None, verbose=False, **model_kws):
        # Init
        base_model_name = str(base_model)[:-2].split(".")[-1]
        self.name = name
        if prefix == "infer":
            if base_model_name == "TSNE":
                prefix = f"t-SNE."
            else:
                prefix = f"{base_model_name}."
        self.prefix = prefix
        self.verbose = verbose

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
                if verbose:
                    print(f"Computing PCA for input data using {self.n_components__pca} components", file=sys.stderr)
                self.X_input = PrincipalComponentAnalysis(X=self.X, base_model=PCA, n_components=self.n_components__pca, **pca_kws).projection_

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
            if self.verbose:
                print(f"Data shape could not support {n_components} and is using the maximum {possible_components} instead.", file=sys.stderr)
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
        if self.verbose:
            print(f"Precomputed: {precomputed}", file=sys.stderr)
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
def eigenprofiles_from_data(X:pd.DataFrame, y:pd.Series, class_type:str = None, mode="eigenprofiles", pca_kws=dict(), verbose:bool=True, axis=0, prefix="PC."):
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
