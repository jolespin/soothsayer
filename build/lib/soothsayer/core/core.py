import os, sys, datetime, copy
from collections import OrderedDict
import pandas as pd
from skbio.util._decorator import experimental, stable

from ..utils import is_path_like, is_nonstring_iterable, pd_dataframe_extend_index, assert_acceptable_arguments
from ..io import read_dataframe, write_object

__all__ = ["Dataset"]
__all__ = sorted(__all__)



# Dataset
@experimental(as_of="2019.06")
class Dataset(object):
    def __init__(self, data:pd.DataFrame, metadata_observations:pd.DataFrame=None, metadata_attributes:pd.DataFrame=None, metadata_target_field=None, name=None, description=None, obsv_type=None, attr_type=None, metric_type=None, name_initial_data=None, check_index_overlap=True, alias_metadata_observations:str="m0", alias_metadata_attributes:str="m1", **additional_fields):
        """
        Dataset class to store different versions of a dataset.  Currently only supports 2D pd.DataFrames but will support ND xarrays in the future
        Assumes columns/axis=1 contains attributes and index/axis=0 contains observations for compatibility with scikit-learn

        Usage:
        from soothsayer import Dataset
        from soothsayer.transmute import normalize

        ds_iris = Dataset(X_iris, name="Iris", metadata_observations=y_iris, description="iris dataset", obsv_type="iris samples", attr_type="leaf lengths", metric_type="cm")
        ds_iris.add_version(name_version="zscore", data=normalize(X_iris, "zscore", axis=0), notes="normalized across axis=0")
        ds_iris.set_metadata_target("Species")
        ds_iris.add_indexing_subset("versicolor", ds_iris.y[lambda x: x == "virginica"].index, axis=0)
        ds_iris.add_indexing_subset("sepal", ['sepal_length', 'sepal_width'], axis=1)

        # Initial dataset
        X, y = ds_iris.get_dataset(None, return_X_y=True)
        print("Initial:", X.shape)
        # Initial: (150, 4)

        # Versicolor dataset
        X, y = ds_iris.get_dataset(None, observation_subset="versicolor", return_X_y=True)
        print("Versicolor:", X.shape)
        # Versicolor: (50, 4)

        # Versicolor dataset with sepal
        X, y = ds_iris.get_dataset(None, observation_subset="versicolor", attribute_subset="sepal", return_X_y=True)
        print("Versicolor only sepal:", X.shape)
        # Versicolor only sepal: (50, 2)
        """
        # Labels
        self.name = name
        self.description = description
        self.obsv_type = obsv_type
        self.attr_type = attr_type
        self.metric_type = metric_type
        self.name_initial_data = name_initial_data

        # Subsets
        self.attribute_subsets = OrderedDict()
        self.observation_subsets = OrderedDict()
        self.check_index_overlap = check_index_overlap

        # Initialize
        self.number_of_metadata_observations_fields = 0
        self.number_of_metadata_attributes_fields = 0
        self.metadata_observations = None
        self.alias_metadata_observations = alias_metadata_observations
        self.metadata_attributes = None
        self.alias_metadata_attributes = alias_metadata_attributes
        self.y_field = None
        self.y = None
        self.__synthesized__ = datetime.datetime.utcnow()
        self.__database__ = OrderedDict()
        if is_path_like(data, path_must_exist=True):
            data = read_dataframe(data)

        self.add_version(name_version=name_initial_data, data=data, **additional_fields)
        self.add_indexing_subset(name_subset=None, labels=data.index, axis=0)
        self.add_indexing_subset(name_subset=None, labels=data.columns, axis=1)
        self.set_default( name_version=name_initial_data, observation_subset=None, attribute_subset=None)

        # Metadata observations
        if metadata_observations is None:
            metadata_observations = pd_dataframe_extend_index(data.index, pd.DataFrame(), axis=0)
        self.add_metadata(metadata_observations, axis="observations", metadata_target_field=metadata_target_field)



        # Metadata attributes
        if metadata_attributes is None:
            metadata_attributes = pd_dataframe_extend_index(data.columns, pd.DataFrame(), axis=0)
        self.add_metadata(metadata_attributes, axis="attributes", metadata_target_field=None)


    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1][:-2]
        lines = [f"Dataset| {self.name} | {self.__database__[self.name_initial_data]['shape']}"]
        lines.append(len(lines[0])*"=")
        lines.append("\n".join([
            f"        obsv_type: {self.obsv_type}",
            f"        attr_type: {self.attr_type}",
            f"        metric_type: {self.metric_type}",
            f"        description: {self.description}",
        ]))
        lines.append("\n".join([
            f"datasets: {list(self.__database__.keys())}",
            f"attribute_subsets: {list(self.attribute_subsets.keys())}",
            f"observation_subsets: {list(self.observation_subsets.keys())}",
        ]))
        lines.append("\n".join([
                    f"        metadata_observations: {self.number_of_metadata_observations_fields}",
                    f"        metadata_attributes: {self.number_of_metadata_attributes_fields}"
        ]))
        lines.append(f"        default: {self.X_version} | {self.X.shape}")
        lines.append("        " + self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S"))
        return "\n".join(lines)

    # Add metadata
    def add_metadata(self, metadata:pd.DataFrame, axis="infer", metadata_target_field=None):
        accepted_axis = {"attrs", "attributes","columns", 1, "obsvs", "observations","rows", 0}
        assert metadata is not None, "metadata can't be None"
        if is_path_like(metadata, path_must_exist=True):
            metadata = read_dataframe(metadata)
        initial_data_observations = self.get_indexing_subset(name_subset=None, axis="observations")
        initial_data_attributes = self.get_indexing_subset(name_subset=None, axis="attributes")

        # Convert pd.Series to pd.DataFrame
        if isinstance(metadata, pd.Series):
            metadata = metadata.to_frame()

        # Check the axis
        if axis == "infer":
            if len(set(initial_data_observations) & len(metadata.index)) > 0:
                axis = "observations"
            if len(set(initial_data_attributes) & len(metadata.index)) > 0:
                axis = "attributes"
            assert axis != "infer", "Please check the indicies of the metadata and how they match against the raw initial data"
        assert axis in accepted_axis, "axis must be one of the following: {accepted_axis}"
        # Metadata observations
        if axis in { "obsvs", "observations","rows", 0}:
            if self.check_index_overlap:
                assert set(metadata.index) >= set(initial_data_observations), "Not all index of metadata are in initial data"

            self.metadata_observations = metadata.copy()
            self.number_of_metadata_observations_fields = self.metadata_observations.shape[1]
            if isinstance(self.metadata_observations, pd.Series):
                self.metadata_observations = self.metadata_observations.to_frame()
            if self.metadata_observations.shape[1] == 1:
                metadata_target_field = self.metadata_observations.columns[0]

            # Get useable subset
            if self.check_index_overlap:
                self.metadata_observations = self.metadata_observations.loc[initial_data_observations]
            if metadata_target_field is None:
                self.y_field = None
                self.y = None
            if metadata_target_field is not None:
                assert metadata_target_field in self.metadata_observations.columns
                self.y_field = metadata_target_field
                self.y = self.metadata_observations[self.y_field]

            if self.alias_metadata_observations is not None:
                setattr(self, str(self.alias_metadata_observations), self.metadata_observations)

        # Metadata attributes
        if axis in {"attrs", "attributes","columns", 1}:
            if self.check_index_overlap:
                assert set(metadata.index) >= set(initial_data_attributes), "Not all index of metadata are in initial data"

            self.metadata_attributes = metadata.copy()
            self.number_of_metadata_attributes_fields = self.metadata_attributes.shape[1]
            if isinstance(self.metadata_attributes, pd.Series):
                self.metadata_attributes = self.metadata_attributes.to_frame()
            if self.check_index_overlap:
                self.metadata_attributes = self.metadata_attributes.loc[initial_data_attributes]
            if self.alias_metadata_attributes is not None:
                setattr(self, str(self.alias_metadata_attributes), self.metadata_attributes)
        return self

    # Add data versions
    def add_version(self, name_version, data:pd.DataFrame, alias=None, notes=None,**additional_fields):
        """
        Add a version of the data.  For example, a version of data with a different transformation
        """
        self.__database__[name_version] = {"data":data.copy(), "shape":data.shape, "notes":notes, "alias":alias, **additional_fields}
        if alias is not None:
            setattr(self, str(alias), self.__database__[name_version]["data"])
        return self

    # Add indexing subsets for observations and attributes
    def add_indexing_subset(self, name_subset, labels, axis, alias=None, notes=None, **additional_fields):
        """
        Add an indexing subset.  For example, a feature selection subset or a filtered observation subset
        """
        accepted_axis = {"attrs", "attributes","columns", 1, "obsvs", "observations","rows", 0}
        assert axis in accepted_axis, f"`axis` must be one of the following: {accepted_axis}"

        labels = pd.Index(labels, name=name_subset)
        if axis in { "obsvs", "observations","rows", 0}:
            assert set(labels) <= set(self.get_dataset(name_version=self.name_initial_data).index), f"Not all labels are in the {self.name_initial_data} index"
            self.observation_subsets[name_subset] = {"labels":labels, "notes":notes, "alias":alias, **additional_fields}

        if axis in { "attrs", "attributes","columns", 1}:
            assert set(labels) <= set(self.get_dataset(name_version=self.name_initial_data).columns), f"Not all labels are in the {self.name_initial_data} columns"
            self.attribute_subsets[name_subset] = {"labels":labels, "notes":notes, "alias":alias, **additional_fields}
        if alias is not None:
            setattr(self, str(alias), labels)


    # Get indexing subset
    def get_indexing_subset(self, name_subset=None, axis=0):
        accepted_axis = {"attrs", "attributes","columns", 1, "obsvs", "observations","rows", 0}
        assert axis in accepted_axis, f"`axis` must be one of the following: {accepted_axis}"
        if axis in { "obsvs", "observations","rows", 0}:
            return self.observation_subsets[name_subset]["labels"]
        if axis in { "attrs", "attributes","columns", 1}:
            return self.attribute_subsets[name_subset]["labels"]
    # Get item wrapper
    def __getitem__(self, key):
        return self.get_dataset(name_version=key, return_X_y=False, observation_subset=None, attribute_subset=None)

    # Get stored datasets
    def get_dataset(self, name_version, return_X_y=False, observation_subset=None, attribute_subset=None):
        """
        Retrieve a dataset that has been stored
        """
         # Internal helper function to get the indices
        def _get_index(df, subset, axis):
            # Default
            if subset is None:
                 if axis == 0:
                     return df.index
                 if axis == 1:
                     return df.columns
            # Function
            if hasattr(subset, "__call__"):
                 if axis == 0:
                     return [*filter(subset, df.index)]
                 if axis == 1:
                     return [*filter(subset, df.columns)]
            # Custom list-like
            if is_nonstring_iterable(subset):
                 return subset

            # Collection
            if subset in set(self.observation_subsets.keys()) | set(self.attribute_subsets.keys()):
                if axis == 0:
                    return self.observation_subsets[subset]["labels"]
                if axis == 1:
                    return self.attribute_subsets[subset]["labels"]
            raise ValueError("`subset` must either be a a key in `self.[axis]_subsets`, a list-like object, or a function to filter")


        # If no version is specified then use the default
        assert name_version in self.__database__, f"Cannot find `{name_version}`.  Please add it to the datasets via `add_version`"

        # Data
        df = self.__database__[name_version]["data"]
         # Observations
        idx_obsvs = _get_index(df=df, subset=observation_subset, axis=0)
        assert set(idx_obsvs) <= set(df.index), f"Not all labels are in the {name_version} index"
        # Attributes
        idx_attrs = _get_index(df=df, subset=attribute_subset, axis=1)
        assert set(idx_attrs) <= set(df.columns), f"Not all labels are in the {name_version} columns"

        X = df.loc[idx_obsvs, idx_attrs]
        if return_X_y:
            assert self.metadata_observations is not None, "`metadata_observations` cannot be None"
            assert self.y_field is not None, "Please set `y_field` with `set_metadata_target`"
            return X, self.y[idx_obsvs]
        else:
            return X
    # Get dataset field
    def get_dataset_field(self, name_version=None, field=None):
        if (name_version is not None) and (field is None):
            print(f"Fields available for Dataset({name_version}):\n", set(self.__database__[name_version].keys()), sep="\t", file=sys.stdout)
        if (name_version is None) and (field is None):
            print("Available options:", file=sys.stdout)
            for k,d in self.__database__.items():
                print(f"\t{k}:", set(d.keys()), sep="\t", file=sys.stdout)
        assert name_version in self.__database__, f"{name_version} is not available.  Please use `add_version`."
        assert field in self.__database__[name_version], f"{field} not available for dataset version: {name_version}"
        return self.__database__[name_version][field]

    # Get indexing field
    def get_indexing_field(self, name_subset=None, field=None, axis=None):
        if axis is None:
            if (name_subset in self.observation_subsets) and (name_subset not in self.attribute_subsets):
                axis = 0
            if (name_subset in self.attribute_subsets) and (name_subset not in self.observation_subsets):
                axis = 1
            assert axis is not None, f"Cannot infer axis.  Please specify axis or add the version."
            print(f"Inferring axis to be {axis}", file=sys.stderr)

        accepted_axis = {"attrs", "attributes","columns", 1, "obsvs", "observations","rows", 0}
        assert axis in accepted_axis, f"`axis` must be one of the following: {accepted_axis}"
        if axis in { "obsvs", "observations","rows", 0}:
            assert name_subset in self.observation_subsets, f"{name_subset} not available for axis={axis}"
            assert field in self.observation_subsets[name_subset], f"{field} not available for indexing subset: {name_subset}"
            return self.observation_subsets[name_subset][field]
        if axis in { "attrs", "attributes","columns", 1}:
            assert name_subset in self.attribute_subsets, f"{name_subset} not available for axis={axis}"
            assert field in self.attribute_subsets[name_subset], f"{field} not available for indexing subset: {name_subset}"
            return self.attribute_subsets[name_subset][field]

    # Set metadata target vector
    def set_metadata_target(self, field):
        """
        metadata_observations
        """
        assert self.metadata_observations is not None, "No `metadata_observations` available"
        assert field in self.metadata_observations.columns, f"{field} not in `metadata_observations`` columns"
        self.y_field = field
        self.y = self.metadata_observations[field]
        return self

    # Set defaults
    def set_default(self, name_version, observation_subset=None, attribute_subset=None):
        assert name_version in self.__database__, f"Cannot find `{name_version}`.  Please add it to the datasets via `add_version`"
        assert observation_subset in self.observation_subsets, f"Cannot find `{observation_subset}`.  Please add it to the datasets via `add_indexing_subset`"
        assert attribute_subset in self.attribute_subsets, f"Cannot find `{attribute_subset}`.  Please add it to the datasets via `add_indexing_subset`"

        self.X_version = name_version
        self.X = self.get_dataset(name_version=name_version, return_X_y=False, observation_subset=observation_subset, attribute_subset=attribute_subset)
        self.index = self.X.index
        self.index_version = observation_subset
        self.columns = self.X.columns
        self.columns_version = attribute_subset
        return self

#     # Filter dataset
#     def filter(self, func_observations=None, func_attributes=None, name_version=None):
#         """
#         Filter a datasets

#         #! Revisit this
#         """
#         # If no version is specified then use the default
#         if name_version is None:
#             name_version = self.X_version
#         assert name_version in self.__database__, f"Cannot find `{name_version}`.  Please add it to the datasets via `add_version`"
#         df = self.__database__[name_version]["data"]
#         # Observations
#         idx_observations = df.index
#         if func_observations is not None:
#             idx_observations = [*filter(func_observations, idx_observations)]
#         # Attributes
#         idx_attributes = df.columns
#         if func_attributes is not None:
#             idx_attributes = [*filter(func_attributes, idx_attributes)]
#         return df.loc[idx_observations, idx_attributes]

    # Write object to file
    def to_file(self, path:str, compression="infer"):
        write_object(self, path=path, compression=compression)
        return self

    # Built-In
    def __setitem__(self, name_version, data):
        self.add_version(name_version, data)

    def __getitem__(self, name_version):
        return self.__database__[name_version]["data"]

    def __len__(self):
        return len(self.__database__)

    def __delitem__(self, name_version):
        del self.__database__[name_version]

    def clear(self):
        return self.__database__.clear()

    def copy(self):
        return copy.deepcopy(self)

    def has_key(self, k):
        return k in self.__database__

    def update(self, *args, **kwargs):
        return self.__database__.update(*args, **kwargs)

    def keys(self):
        print(f"Dataset versions:")
        return list(self.__database__.keys())

    def values(self):
        print(f"Datasets:")
        return list(self.__database__.values())

    def items(self):
        return self.__database__.items()

    def __contains__(self, item):
        return item in self.__database__

    def __iter__(self):
        for name_version, d in self.__database__.items():
            yield name_version, d["data"]

    def __call__(self, field, index=None, func_filter=None, func_map=None, axis=0):
        assert_acceptable_arguments(axis, {0,1})
        assert not is_nonstring_iterable(field), "`field` cannot be a non-string iterable"
        if axis == 0:
            assert self.metadata_observations is not None
            assert field in self.metadata_observations.columns, "`{}` not in `metadata_observations`".format(field)
            data = self.metadata_observations[field]
        if axis == 1:
            assert self.metadata_attributes is not None
            assert field in self.metadata_attributes.columns, "`{}` not in `metadata_attributes`".format(field)
            data = self.metadata_attributes[field]
        if index is not None:
            data = data[index]
        if func_filter is not None:
            data = data[func_filter]
        if func_map is not None:
            data = data.map(func_map)

        return data


    def copy(self):
        return copy.deepcopy(self)
