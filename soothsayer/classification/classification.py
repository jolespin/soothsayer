# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, copy, warnings
from collections import OrderedDict, defaultdict
from tqdm import tqdm

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# SciPy
from scipy import stats

# Scikit-Learn
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import ete3

# Soothsayer
from ..utils import *
from ..io import read_object, write_object
from ..visuals import draw_networkx_labels_with_box, OutlineCollection

__all__ = ["HierarchicalClassifier", "get_mixed_feature_pipeline_estimator"]
__all__ = sorted(__all__)

# Get mixed feature pipeline estimator
def get_mixed_feature_pipeline_estimator(
    estimators:dict,
    features:dict,
    ):
    """
    # Prepare data with multi-level columns
    X_sepal = pd.DataFrame({'sepal_length': {'iris_0': 5.1,'iris_1': 4.9,'iris_2': 4.7,'iris_3': 4.6,'iris_4': 5.0,'iris_5': 5.4,'iris_6': 4.6,'iris_7': 5.0,'iris_8': 4.4,'iris_9': 4.9,'iris_10': 5.4,'iris_11': 4.8,'iris_12': 4.8,'iris_13': 4.3,'iris_14': 5.8,'iris_15': 5.7,'iris_16': 5.4,'iris_17': 5.1,'iris_18': 5.7,'iris_19': 5.1,'iris_20': 5.4,'iris_21': 5.1,'iris_22': 4.6,'iris_23': 5.1,'iris_24': 4.8,'iris_25': 5.0,'iris_26': 5.0,'iris_27': 5.2,'iris_28': 5.2,'iris_29': 4.7,'iris_30': 4.8,'iris_31': 5.4,'iris_32': 5.2,'iris_33': 5.5,'iris_34': 4.9,'iris_35': 5.0,'iris_36': 5.5,'iris_37': 4.9,'iris_38': 4.4,'iris_39': 5.1,'iris_40': 5.0,'iris_41': 4.5,'iris_42': 4.4,'iris_43': 5.0,'iris_44': 5.1,'iris_45': 4.8,'iris_46': 5.1,'iris_47': 4.6,'iris_48': 5.3,'iris_49': 5.0,'iris_50': 7.0,'iris_51': 6.4,'iris_52': 6.9,'iris_53': 5.5,'iris_54': 6.5,'iris_55': 5.7,'iris_56': 6.3,'iris_57': 4.9,'iris_58': 6.6,'iris_59': 5.2,'iris_60': 5.0,'iris_61': 5.9,'iris_62': 6.0,'iris_63': 6.1,'iris_64': 5.6,'iris_65': 6.7,'iris_66': 5.6,'iris_67': 5.8,'iris_68': 6.2,'iris_69': 5.6,'iris_70': 5.9,'iris_71': 6.1,'iris_72': 6.3,'iris_73': 6.1,'iris_74': 6.4,'iris_75': 6.6,'iris_76': 6.8,'iris_77': 6.7,'iris_78': 6.0,'iris_79': 5.7,'iris_80': 5.5,'iris_81': 5.5,'iris_82': 5.8,'iris_83': 6.0,'iris_84': 5.4,'iris_85': 6.0,'iris_86': 6.7,'iris_87': 6.3,'iris_88': 5.6,'iris_89': 5.5,'iris_90': 5.5,'iris_91': 6.1,'iris_92': 5.8,'iris_93': 5.0,'iris_94': 5.6,'iris_95': 5.7,'iris_96': 5.7,'iris_97': 6.2,'iris_98': 5.1,'iris_99': 5.7,'iris_100': 6.3,'iris_101': 5.8,'iris_102': 7.1,'iris_103': 6.3,'iris_104': 6.5,'iris_105': 7.6,'iris_106': 4.9,'iris_107': 7.3,'iris_108': 6.7,'iris_109': 7.2,'iris_110': 6.5,'iris_111': 6.4,'iris_112': 6.8,'iris_113': 5.7,'iris_114': 5.8,'iris_115': 6.4,'iris_116': 6.5,'iris_117': 7.7,'iris_118': 7.7,'iris_119': 6.0,'iris_120': 6.9,'iris_121': 5.6,'iris_122': 7.7,'iris_123': 6.3,'iris_124': 6.7,'iris_125': 7.2,'iris_126': 6.2,'iris_127': 6.1,'iris_128': 6.4,'iris_129': 7.2,'iris_130': 7.4,'iris_131': 7.9,'iris_132': 6.4,'iris_133': 6.3,'iris_134': 6.1,'iris_135': 7.7,'iris_136': 6.3,'iris_137': 6.4,'iris_138': 6.0,'iris_139': 6.9,'iris_140': 6.7,'iris_141': 6.9,'iris_142': 5.8,'iris_143': 6.8,'iris_144': 6.7,'iris_145': 6.7,'iris_146': 6.3,'iris_147': 6.5,'iris_148': 6.2,'iris_149': 5.9},'sepal_width': {'iris_0': 3.5,'iris_1': 3.0,'iris_2': 3.2,'iris_3': 3.1,'iris_4': 3.6,'iris_5': 3.9,'iris_6': 3.4,'iris_7': 3.4,'iris_8': 2.9,'iris_9': 3.1,'iris_10': 3.7,'iris_11': 3.4,'iris_12': 3.0,'iris_13': 3.0,'iris_14': 4.0,'iris_15': 4.4,'iris_16': 3.9,'iris_17': 3.5,'iris_18': 3.8,'iris_19': 3.8,'iris_20': 3.4,'iris_21': 3.7,'iris_22': 3.6,'iris_23': 3.3,'iris_24': 3.4,'iris_25': 3.0,'iris_26': 3.4,'iris_27': 3.5,'iris_28': 3.4,'iris_29': 3.2,'iris_30': 3.1,'iris_31': 3.4,'iris_32': 4.1,'iris_33': 4.2,'iris_34': 3.1,'iris_35': 3.2,'iris_36': 3.5,'iris_37': 3.6,'iris_38': 3.0,'iris_39': 3.4,'iris_40': 3.5,'iris_41': 2.3,'iris_42': 3.2,'iris_43': 3.5,'iris_44': 3.8,'iris_45': 3.0,'iris_46': 3.8,'iris_47': 3.2,'iris_48': 3.7,'iris_49': 3.3,'iris_50': 3.2,'iris_51': 3.2,'iris_52': 3.1,'iris_53': 2.3,'iris_54': 2.8,'iris_55': 2.8,'iris_56': 3.3,'iris_57': 2.4,'iris_58': 2.9,'iris_59': 2.7,'iris_60': 2.0,'iris_61': 3.0,'iris_62': 2.2,'iris_63': 2.9,'iris_64': 2.9,'iris_65': 3.1,'iris_66': 3.0,'iris_67': 2.7,'iris_68': 2.2,'iris_69': 2.5,'iris_70': 3.2,'iris_71': 2.8,'iris_72': 2.5,'iris_73': 2.8,'iris_74': 2.9,'iris_75': 3.0,'iris_76': 2.8,'iris_77': 3.0,'iris_78': 2.9,'iris_79': 2.6,'iris_80': 2.4,'iris_81': 2.4,'iris_82': 2.7,'iris_83': 2.7,'iris_84': 3.0,'iris_85': 3.4,'iris_86': 3.1,'iris_87': 2.3,'iris_88': 3.0,'iris_89': 2.5,'iris_90': 2.6,'iris_91': 3.0,'iris_92': 2.6,'iris_93': 2.3,'iris_94': 2.7,'iris_95': 3.0,'iris_96': 2.9,'iris_97': 2.9,'iris_98': 2.5,'iris_99': 2.8,'iris_100': 3.3,'iris_101': 2.7,'iris_102': 3.0,'iris_103': 2.9,'iris_104': 3.0,'iris_105': 3.0,'iris_106': 2.5,'iris_107': 2.9,'iris_108': 2.5,'iris_109': 3.6,'iris_110': 3.2,'iris_111': 2.7,'iris_112': 3.0,'iris_113': 2.5,'iris_114': 2.8,'iris_115': 3.2,'iris_116': 3.0,'iris_117': 3.8,'iris_118': 2.6,'iris_119': 2.2,'iris_120': 3.2,'iris_121': 2.8,'iris_122': 2.8,'iris_123': 2.7,'iris_124': 3.3,'iris_125': 3.2,'iris_126': 2.8,'iris_127': 3.0,'iris_128': 2.8,'iris_129': 3.0,'iris_130': 2.8,'iris_131': 3.8,'iris_132': 2.8,'iris_133': 2.8,'iris_134': 2.6,'iris_135': 3.0,'iris_136': 3.4,'iris_137': 3.1,'iris_138': 3.0,'iris_139': 3.1,'iris_140': 3.1,'iris_141': 3.1,'iris_142': 2.7,'iris_143': 3.2,'iris_144': 3.3,'iris_145': 3.0,'iris_146': 2.5,'iris_147': 3.0,'iris_148': 3.4,'iris_149': 3.0}})
    X_petal = pd.DataFrame({'petal_length': {'iris_0': 1.4,'iris_1': 1.4,'iris_2': 1.3,'iris_3': 1.5,'iris_4': 1.4,'iris_5': 1.7,'iris_6': 1.4,'iris_7': 1.5,'iris_8': 1.4,'iris_9': 1.5,'iris_10': 1.5,'iris_11': 1.6,'iris_12': 1.4,'iris_13': 1.1,'iris_14': 1.2,'iris_15': 1.5,'iris_16': 1.3,'iris_17': 1.4,'iris_18': 1.7,'iris_19': 1.5,'iris_20': 1.7,'iris_21': 1.5,'iris_22': 1.0,'iris_23': 1.7,'iris_24': 1.9,'iris_25': 1.6,'iris_26': 1.6,'iris_27': 1.5,'iris_28': 1.4,'iris_29': 1.6,'iris_30': 1.6,'iris_31': 1.5,'iris_32': 1.5,'iris_33': 1.4,'iris_34': 1.5,'iris_35': 1.2,'iris_36': 1.3,'iris_37': 1.4,'iris_38': 1.3,'iris_39': 1.5,'iris_40': 1.3,'iris_41': 1.3,'iris_42': 1.3,'iris_43': 1.6,'iris_44': 1.9,'iris_45': 1.4,'iris_46': 1.6,'iris_47': 1.4,'iris_48': 1.5,'iris_49': 1.4,'iris_50': 4.7,'iris_51': 4.5,'iris_52': 4.9,'iris_53': 4.0,'iris_54': 4.6,'iris_55': 4.5,'iris_56': 4.7,'iris_57': 3.3,'iris_58': 4.6,'iris_59': 3.9,'iris_60': 3.5,'iris_61': 4.2,'iris_62': 4.0,'iris_63': 4.7,'iris_64': 3.6,'iris_65': 4.4,'iris_66': 4.5,'iris_67': 4.1,'iris_68': 4.5,'iris_69': 3.9,'iris_70': 4.8,'iris_71': 4.0,'iris_72': 4.9,'iris_73': 4.7,'iris_74': 4.3,'iris_75': 4.4,'iris_76': 4.8,'iris_77': 5.0,'iris_78': 4.5,'iris_79': 3.5,'iris_80': 3.8,'iris_81': 3.7,'iris_82': 3.9,'iris_83': 5.1,'iris_84': 4.5,'iris_85': 4.5,'iris_86': 4.7,'iris_87': 4.4,'iris_88': 4.1,'iris_89': 4.0,'iris_90': 4.4,'iris_91': 4.6,'iris_92': 4.0,'iris_93': 3.3,'iris_94': 4.2,'iris_95': 4.2,'iris_96': 4.2,'iris_97': 4.3,'iris_98': 3.0,'iris_99': 4.1,'iris_100': 6.0,'iris_101': 5.1,'iris_102': 5.9,'iris_103': 5.6,'iris_104': 5.8,'iris_105': 6.6,'iris_106': 4.5,'iris_107': 6.3,'iris_108': 5.8,'iris_109': 6.1,'iris_110': 5.1,'iris_111': 5.3,'iris_112': 5.5,'iris_113': 5.0,'iris_114': 5.1,'iris_115': 5.3,'iris_116': 5.5,'iris_117': 6.7,'iris_118': 6.9,'iris_119': 5.0,'iris_120': 5.7,'iris_121': 4.9,'iris_122': 6.7,'iris_123': 4.9,'iris_124': 5.7,'iris_125': 6.0,'iris_126': 4.8,'iris_127': 4.9,'iris_128': 5.6,'iris_129': 5.8,'iris_130': 6.1,'iris_131': 6.4,'iris_132': 5.6,'iris_133': 5.1,'iris_134': 5.6,'iris_135': 6.1,'iris_136': 5.6,'iris_137': 5.5,'iris_138': 4.8,'iris_139': 5.4,'iris_140': 5.6,'iris_141': 5.1,'iris_142': 5.1,'iris_143': 5.9,'iris_144': 5.7,'iris_145': 5.2,'iris_146': 5.0,'iris_147': 5.2,'iris_148': 5.4,'iris_149': 5.1},'petal_width': {'iris_0': 0.2,'iris_1': 0.2,'iris_2': 0.2,'iris_3': 0.2,'iris_4': 0.2,'iris_5': 0.4,'iris_6': 0.3,'iris_7': 0.2,'iris_8': 0.2,'iris_9': 0.1,'iris_10': 0.2,'iris_11': 0.2,'iris_12': 0.1,'iris_13': 0.1,'iris_14': 0.2,'iris_15': 0.4,'iris_16': 0.4,'iris_17': 0.3,'iris_18': 0.3,'iris_19': 0.3,'iris_20': 0.2,'iris_21': 0.4,'iris_22': 0.2,'iris_23': 0.5,'iris_24': 0.2,'iris_25': 0.2,'iris_26': 0.4,'iris_27': 0.2,'iris_28': 0.2,'iris_29': 0.2,'iris_30': 0.2,'iris_31': 0.4,'iris_32': 0.1,'iris_33': 0.2,'iris_34': 0.2,'iris_35': 0.2,'iris_36': 0.2,'iris_37': 0.1,'iris_38': 0.2,'iris_39': 0.2,'iris_40': 0.3,'iris_41': 0.3,'iris_42': 0.2,'iris_43': 0.6,'iris_44': 0.4,'iris_45': 0.3,'iris_46': 0.2,'iris_47': 0.2,'iris_48': 0.2,'iris_49': 0.2,'iris_50': 1.4,'iris_51': 1.5,'iris_52': 1.5,'iris_53': 1.3,'iris_54': 1.5,'iris_55': 1.3,'iris_56': 1.6,'iris_57': 1.0,'iris_58': 1.3,'iris_59': 1.4,'iris_60': 1.0,'iris_61': 1.5,'iris_62': 1.0,'iris_63': 1.4,'iris_64': 1.3,'iris_65': 1.4,'iris_66': 1.5,'iris_67': 1.0,'iris_68': 1.5,'iris_69': 1.1,'iris_70': 1.8,'iris_71': 1.3,'iris_72': 1.5,'iris_73': 1.2,'iris_74': 1.3,'iris_75': 1.4,'iris_76': 1.4,'iris_77': 1.7,'iris_78': 1.5,'iris_79': 1.0,'iris_80': 1.1,'iris_81': 1.0,'iris_82': 1.2,'iris_83': 1.6,'iris_84': 1.5,'iris_85': 1.6,'iris_86': 1.5,'iris_87': 1.3,'iris_88': 1.3,'iris_89': 1.3,'iris_90': 1.2,'iris_91': 1.4,'iris_92': 1.2,'iris_93': 1.0,'iris_94': 1.3,'iris_95': 1.2,'iris_96': 1.3,'iris_97': 1.3,'iris_98': 1.1,'iris_99': 1.3,'iris_100': 2.5,'iris_101': 1.9,'iris_102': 2.1,'iris_103': 1.8,'iris_104': 2.2,'iris_105': 2.1,'iris_106': 1.7,'iris_107': 1.8,'iris_108': 1.8,'iris_109': 2.5,'iris_110': 2.0,'iris_111': 1.9,'iris_112': 2.1,'iris_113': 2.0,'iris_114': 2.4,'iris_115': 2.3,'iris_116': 1.8,'iris_117': 2.2,'iris_118': 2.3,'iris_119': 1.5,'iris_120': 2.3,'iris_121': 2.0,'iris_122': 2.0,'iris_123': 1.8,'iris_124': 2.1,'iris_125': 1.8,'iris_126': 1.8,'iris_127': 1.8,'iris_128': 2.1,'iris_129': 1.6,'iris_130': 1.9,'iris_131': 2.0,'iris_132': 2.2,'iris_133': 1.5,'iris_134': 1.4,'iris_135': 2.3,'iris_136': 2.4,'iris_137': 1.8,'iris_138': 1.8,'iris_139': 2.1,'iris_140': 2.4,'iris_141': 2.3,'iris_142': 1.9,'iris_143': 2.3,'iris_144': 2.5,'iris_145': 2.3,'iris_146': 1.9,'iris_147': 2.0,'iris_148': 2.3,'iris_149': 1.8}})
    y_iris = pd.Series({'iris_0': 'setosa','iris_1': 'setosa','iris_2': 'setosa','iris_3': 'setosa','iris_4': 'setosa','iris_5': 'setosa','iris_6': 'setosa','iris_7': 'setosa','iris_8': 'setosa','iris_9': 'setosa','iris_10': 'setosa','iris_11': 'setosa','iris_12': 'setosa','iris_13': 'setosa','iris_14': 'setosa','iris_15': 'setosa','iris_16': 'setosa','iris_17': 'setosa','iris_18': 'setosa','iris_19': 'setosa','iris_20': 'setosa','iris_21': 'setosa','iris_22': 'setosa','iris_23': 'setosa','iris_24': 'setosa','iris_25': 'setosa','iris_26': 'setosa','iris_27': 'setosa','iris_28': 'setosa','iris_29': 'setosa','iris_30': 'setosa','iris_31': 'setosa','iris_32': 'setosa','iris_33': 'setosa','iris_34': 'setosa','iris_35': 'setosa','iris_36': 'setosa','iris_37': 'setosa','iris_38': 'setosa','iris_39': 'setosa','iris_40': 'setosa','iris_41': 'setosa','iris_42': 'setosa','iris_43': 'setosa','iris_44': 'setosa','iris_45': 'setosa','iris_46': 'setosa','iris_47': 'setosa','iris_48': 'setosa','iris_49': 'setosa','iris_50': 'versicolor','iris_51': 'versicolor','iris_52': 'versicolor','iris_53': 'versicolor','iris_54': 'versicolor','iris_55': 'versicolor','iris_56': 'versicolor','iris_57': 'versicolor','iris_58': 'versicolor','iris_59': 'versicolor','iris_60': 'versicolor','iris_61': 'versicolor','iris_62': 'versicolor','iris_63': 'versicolor','iris_64': 'versicolor','iris_65': 'versicolor','iris_66': 'versicolor','iris_67': 'versicolor','iris_68': 'versicolor','iris_69': 'versicolor','iris_70': 'versicolor','iris_71': 'versicolor','iris_72': 'versicolor','iris_73': 'versicolor','iris_74': 'versicolor','iris_75': 'versicolor','iris_76': 'versicolor','iris_77': 'versicolor','iris_78': 'versicolor','iris_79': 'versicolor','iris_80': 'versicolor','iris_81': 'versicolor','iris_82': 'versicolor','iris_83': 'versicolor','iris_84': 'versicolor','iris_85': 'versicolor','iris_86': 'versicolor','iris_87': 'versicolor','iris_88': 'versicolor','iris_89': 'versicolor','iris_90': 'versicolor','iris_91': 'versicolor','iris_92': 'versicolor','iris_93': 'versicolor','iris_94': 'versicolor','iris_95': 'versicolor','iris_96': 'versicolor','iris_97': 'versicolor','iris_98': 'versicolor','iris_99': 'versicolor','iris_100': 'virginica','iris_101': 'virginica','iris_102': 'virginica','iris_103': 'virginica','iris_104': 'virginica','iris_105': 'virginica','iris_106': 'virginica','iris_107': 'virginica','iris_108': 'virginica','iris_109': 'virginica','iris_110': 'virginica','iris_111': 'virginica','iris_112': 'virginica','iris_113': 'virginica','iris_114': 'virginica','iris_115': 'virginica','iris_116': 'virginica','iris_117': 'virginica','iris_118': 'virginica','iris_119': 'virginica','iris_120': 'virginica','iris_121': 'virginica','iris_122': 'virginica','iris_123': 'virginica','iris_124': 'virginica','iris_125': 'virginica','iris_126': 'virginica','iris_127': 'virginica','iris_128': 'virginica','iris_129': 'virginica','iris_130': 'virginica','iris_131': 'virginica','iris_132': 'virginica','iris_133': 'virginica','iris_134': 'virginica','iris_135': 'virginica','iris_136': 'virginica','iris_137': 'virginica','iris_138': 'virginica','iris_139': 'virginica','iris_140': 'virginica','iris_141': 'virginica','iris_142': 'virginica','iris_143': 'virginica','iris_144': 'virginica','iris_145': 'virginica','iris_146': 'virginica','iris_147': 'virginica','iris_148': 'virginica','iris_149': 'virginica'})
    X_iris = pd.concat([
        sy.utils.pd_prepend_level_to_index(X_sepal, "sepal", axis=1),
        sy.utils.pd_prepend_level_to_index(X_petal, "petal", axis=1),
    ], axis=1)
    
    # Get the estimators
    estimators = get_mixed_feature_pipeline_estimator(
        estimators={"sepal":LogisticRegression(random_state=0), "petal":DecisionTreeClassifier(random_state=0)},
        features={"sepal":X_iris["sepal"].columns, "petal":X_iris["petal"].columns}
    )

    # Create stacking classifier
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=0),
    )

    # Evaluate just like any other model
    X_training, X_testing, y_training, y_testing = train_test_split(X_iris, y_iris, test_size=0.3)
    ensemble.fit(X_training, y_training)
    y_hat = ensemble.predict(X_testing)
    np.mean(y_hat == y_testing)
    # 0.9333333333333333

    """
    assert set(estimators) == set(features), "set(estimators.keys()) must equal set(features.keys())"
    
    pipelines = list()
    for id_estimator, features in features.items():
        pipe = Pipeline([
            ('select', ColumnTransformer([('sel', 'passthrough', list(map(lambda x:(id_estimator, x), features)))], remainder='drop')), 
            ('clf', estimators[id_estimator])
        ])
        pipelines.append((id_estimator, pipe))
    return pipelines

# Hierarchical Classification
class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    copy_X: copy `X` during fitting
    devel[-1]: 2018.11.02
    """
    # Build so these can be stacked
    def __init__(self, name=None, attr_type="attr", class_type="class", obsv_type="input", verbose=True, graph=None, copy_X=True, metadata=dict()):
        # Labels
        self.name = name
        self.attr_type = attr_type
        self.class_type = class_type
        self.obsv_type = obsv_type

        # Graph
        if graph is None:
            graph = nx.OrderedDiGraph()
        self.graph = graph
        if not bool(self.graph.name):
            self.graph.name = name
        self.paths = OrderedDict()
        self.submodels = list()
        self.subgraphs = OrderedDict()
        self.submodel_performance = OrderedDict()

        # Training
        self.training_observations = set()
        self.training_data = dict()

        # Utility
        self.__synthesized__ = datetime.datetime.utcnow()
        self.verbose = verbose
        self._cross_validation_complete = False
        self.copy_X = copy_X

        # Metadata
        self.metadata = dict([
            ("name",name),
            ("attr_type",attr_type),
            ("class_type",class_type),
            ("obsv_type",obsv_type),
            ("synthesized",self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S")),
            ("num_subgraphs",len(self.subgraphs)),
            ("num_training", len(self.training_observations)),
            ("cross_validation_completed", self._cross_validation_complete),
        ]
        )
        self.metadata.update(metadata)


    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.metadata)[1:-1]})"

    def copy(self):
        return copy.deepcopy(self)
    # =======
    # Utility
    # =======

    def _check_cv(self, cv, index):
        """
        The following does a check to make sure that if a cv train/test set is provided then an index with all of the indicies are provided
        """
        assert cv is not None, "`cv` should not be None at this stage.  Check source code for bug"
        if type(cv) != int:
            assert index is not None, "If `cv` is predefined subsets then an ordered`index` has to be provided for future `X` and `y` cross-validation.  Will be updated to attributes during `fit` method"
            maximum_cv_index = max(nx.utils.flatten(cv))
            n, m = len(index), None
            assert  maximum_cv_index <= n, f"Maximum `cv` index is {maximum_cv_index} and exceeds the number of observations in the X which is {n}"
    def _check_fitted(self, clf):
        return hasattr(clf, "classes_")
    # =======
    # Normalization
    # =======
    def _tss(self, x):
        scaling_factor = x.sum()
        return x/scaling_factor
    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=0)
    # ==========
    # Conversion
    # ==========
    # Get the weighted edges from a probability vector
    def _probabilities_to_weighted_edges(self, y_hat):
        """
        predictions is assumed to be either a pd.Series with MultiIndex or dict
        """
        edges = list()
        for target, path in self.paths.items():
            _path = path[1:]
            for i in range(len(_path)-1):
                node_current, node_next = _path[i], _path[i+1]
                proba = y_hat[node_current, node_next]
                edges.append([node_current, node_next, {"weight":proba}])
            edges.append([path[0], path[1], {"weight":1.0}])
        return edges

    # Convert subgraph to prediction
    def _subgraph_to_probabilities(self, subgraph):
        d_edge_weight = OrderedDict()
        for (*edge, d_data) in subgraph.edges(data=True):
            if self.obsv_type not in edge:
                weight = d_data["weight"]
                d_edge_weight[tuple(edge)] = weight
        return pd.Series(list(d_edge_weight.values()), index=pd.MultiIndex.from_tuples(d_edge_weight.keys(), names=["node_A", "node_B"]), name=subgraph.name)

    # Convert prediction to subgraph
    def _probabilities_to_subgraph(self, y_hat):
        edges = self._probabilities_to_weighted_edges(y_hat)
        G = nx.OrderedDiGraph(name=y_hat.name)
        G.add_edges_from(edges)
        return G



    # Save object to file
    def save_model(self, path, compression="infer"):
        warnings.warn("`save_model` is deprecated and will be removed in future versions.  Please use `to_file` instead.", DeprecationWarning)
        write_object(self, path, compression=compression)

    # Save object to file
    def to_file(self, path, compression="infer"):
        write_object(self, path, compression=compression)

    # Convert a NetworkX graph object to an ete3 Tree
    def to_tree(self, subgraph=None, deepcopy=True, into=ete3.Tree):
        """
        Using into = 'default' to do a check to make sure ete3 is installed
        """
        # assert "ete3" in sys.modules, "Cannot convert to tree object because ete3 is not installed"
        is_subgraph = False
        # if into == "default":
            # into = ete3.Tree
        G = subgraph
        if G is not None:
            if G in self.subgraphs:
                G = self.subgraphs[G]
                is_subgraph = True
            if not is_graph(G):
                if self.verbose:
                    print(f"{G} was not in self.subgraphs", file=sys.stderr )
                G = self.graph
        else:
            G = self.graph
        # Add nodes
        subtrees = OrderedDict()
        for node_data in G.nodes(data=deepcopy):
            if deepcopy:
                node, node_attrs = node_data
                subtree = into()
                subtree.name = node # Had to do this becuase ClusterNode doesn't have `name` attribute in args
                subtree.add_fields(**node_attrs)
            else:
                node = node_data
                subtree = into()
                subtree.name = node # Had to do this becuase ClusterNode doesn't have `name` attribute in args
            subtrees[node] = subtree

        # Add edges
        for edge_data in G.edges(data=deepcopy):
            if deepcopy:
                parent, child, attrs = edge_data
                subtrees[parent].add_child(child=subtrees[child])
                subtrees[parent].add_fields(**{(child,k):v for k,v in node_attrs})
                subtrees[child].add_fields(**{(parent,k):v for k,v in node_attrs})
            else:
                parent, child, attrs = edge_data
                subtrees[parent].add_child(child=subtrees[child])
        # Tree rooting
        tree_root = subtrees[self.obsv_type]
        if is_subgraph:
            tree_root.name = G.name
        return tree_root

    # ============
    # Architecture
    # ============

    # Add submodel nodes
    def add_submodel(self, name, clf, attributes, hyperparameters=None, index=None, cv=None, **custom_node_data):
        """
        `custom_node_data` can override 'sub_target', 'hyperparameters', and 'cv'
        """
        # Attributes must be ordered in the way the model was originally fit
        assert all([name is not None, clf is not None, attributes is not None]), "Must provide `name`, `clf`, and `attributes`"
        if hyperparameters is None:
            hyperparameters = clf.get_params()
        # Check to make sure that if the cross-validation is predefined, that index is not None, and all indicies are in index
        if cv is not None:
            self._check_cv(cv, index)
        # Node data organization and override by 'custom_node_data'
        _node_data = dict(hyperparameters=hyperparameters,  X=None, y=None, cv=cv, index=index)
        _node_data.update(custom_node_data)
        # Check to make sure all of the attributes are present in X.columns and the indicies are the same
        if _node_data["X"] and _node_data["y"]:
            X = _node_data["X"]
            y = _nod_data["y"]
            assert set(X.index) == set(y.index), "X.index must have all components in y.index"
            assert set(attributes) <= set(X.columns), "All attributes must be in X.columns"
            if index is not None:
                assert set(index) <= set(X.index), "If `index` is provided with `X` then all of the indicies must be in `X.index`"
        self.graph.add_node(name, clf=clf, attributes=attributes,  **_node_data)
        self.submodels.append(name)
        return self

    # Add paths to graph
    def add_paths(self, paths):
        # What if someone wants to label intermediate paths?
        # How to label paths?
        # Have an attribute that says if it's intermediate or not?
        """
        path is a collection of lists, it will take the final value as the key
        """

        paths = {p[-1]:p for p in paths}
        for label, path in paths.items():
            path = [self.obsv_type] + list(path)
            self.paths[label] = path
            nx.add_path(self.graph, path, name=label)
        root_submodel = set([*map(lambda x:x[1], self.paths.values())])
        assert len(root_submodel) == 1, "There are too many root submodels.  Please make sure that all the nodes start at the same point.  This should be the 2nd note in the path.  e.g., input -> node-1 | then the nodes can branch out.  Try using the `Topology` object and the `get_paths` option for correct formatting."
        self.root_submodel = list(root_submodel)[0]
        return self

    # Add subgraphs from probabilities to model
    def add_subgraphs_from_probabilities(self, df_predict_proba):
        """
        Generates subgraph for each observations predictions and adds to self.subgraphs
        """
        # Nested methods

        # Force pd.DataFrame
        if isinstance(df_predict_proba, pd.Series):
            df_predict_proba = df_predict_proba.to_frame().T

        for id_obsv, y_hat in df_predict_proba.iterrows():
            edges = self._probabilities_to_weighted_edges(y_hat)
            subgraph = nx.OrderedDiGraph(name=id_obsv)
            subgraph.add_edges_from(edges)
            self.subgraphs[id_obsv] = subgraph
        return self

    # Add weights to graph from accuracy scores for each submodel
    def add_weights_from_accuracy(self, Se_accuracy):
        for target, path in self.paths.items():
            _path = path[1:]
            for i in range(len(_path)-1):
                node_current, node_next = _path[i], _path[i+1]
                accuracy = Se_accuracy[node_current]
                self.set_edge_field((node_current,node_next), "weight", accuracy)
            self.set_edge_field((path[0],path[1]), "weight", 1.0)
        return self

    # Extracting and setting graph fields
    def get_submodel_field(self, submodel, field=None):
        if field is not None:
            return self.graph.nodes[submodel][field]
        else:
            return self.graph.nodes[submodel].keys()
    # Setting submodel fields
    def set_submodel_field(self, submodel, field, obj):
        self.graph.nodes[submodel][field] = obj
        return self
    # Setting edge fields
    def set_edge_field(self, edge, field, obj, G=None):
        if G is None:
            G = self.graph
        G[edge[0]][edge[1]][field] = obj
        return self
    # Retrieving submodel node attributes
    def get_attributes(self, submodel=None):
        if submodel is not None:
            return self.get_submodel_field(submodel=submodel, field="attributes")
        else:
            data = [*map(lambda node: self.get_submodel_field(submodel=node, field="attributes"), self.submodels)]
            return pd.Series(data, index=self.submodels, name="attributes")
    # Retrieving submodel node attributes
    def get_classifier(self, submodel=None):
        if submodel is not None:
            return self.get_submodel_field(submodel=submodel, field="clf")
        else:
            data = [*map(lambda node: self.get_submodel_field(submodel=node, field="clf"), self.submodels)]
            return pd.Series(data, index=self.submodels, name="classifier")
    # Retrieve subgraphs
    def get_subgraph(self,subgraph=None):
        if subgraph is not None:
            return self.subgraphs[subgraph]
        else:
            return self.subgraphs

    # Get the cumulative weights from a graph
    def get_cumulative_weights_from_graph(self, G=None):
        if G is None:
            G = self.graph
        edgelist = list()
        for terminus, path in list(self.paths.items()):
            edges = [*map(lambda i:[path[i], path[i+1]], range(0,len(path)-1))]
            weights = np.cumprod([*map(lambda edge: G.edges[edge]["weight"], edges)])
            for e,w in zip(edges,weights):
                ebunch = (*e, {"weight":w})
                edgelist.append(ebunch)
        return edgelist

    # Get target accuracies
    def get_target_scores(self, target=None, name="scores"):
        assert self._cross_validation_complete, f"`self._cross_validation_complete` must be True"
        targets = list(self.paths.keys()) if target is None else [target]
        d_target_score = OrderedDict()
        for target in targets:
            d_target_score[target] = np.product([*map(lambda submodel: self.get_submodel_field(submodel, "accuracy"), list(self.paths[target])[1:-1])])
        return pd.Series(d_target_score, name=name)

    # Determine edge colors for correct and incorrect paths
    def get_edge_colors_accurate_path(self, y_hat, y_true=None, edge_color_correctpath = "#008080", edge_color_incorrectpath="#000000"):
        DEFAULT_edge_color = "#000000"
        if isinstance(y_hat, pd.DataFrame):
            y_hat = y_hat.mean(axis=0)
        if y_true is None:
            assert is_dict_like(y_hat) == False, "If `y_true` is None then `y_hat` must be the name of a training observation"
            assert y_hat in self.training_observations, f"y_hat is {name} and must be in `self.training_observations`"
            y_true = self.training_data["y_terminus"][y_hat]
            y_hat = self._subgraph_to_probabilities(self.subgraphs[y_hat])
        assert is_dict_like(y_hat), "If `y_true` is not None, `y_hat` must be a pd.Series of probabilties"
        assert y_true in self.paths, "`y_true` must be a target/key in `self.paths`"
        y_hat = pd.Series(y_hat)
        G = self._probabilities_to_subgraph(y_hat)
        correct_path = self.paths[y_true]
        correct_edges = [*map(lambda i:(correct_path[i],correct_path[i+1]), range(0,len(correct_path)-1,1))]
        if len(correct_edges) > 0:
            edge_colors = [*map(lambda edge: {True:edge_color_correctpath, False:edge_color_incorrectpath}[(edge in correct_edges) or (edge[0] == self.obsv_type)], G.edges() )]
        else:
            edge_colors = DEFAULT_edge_color
        return edge_colors




    # =======
    # Fitting
    # =======
    def fit(self, X, Y, submodel=None, ignore_submodel_index=False):
        """
        Y should be a matrix of shape (n, k+1) where n= the number of observations and k=the number of submodels with an additional column containing the final prediction class
        Use np.nan if the submodel does not apply for a particular observation
        If type(Y) is pd.Series then `submodel` must be called and will assume `Y` represents a vector of target values for `submodel`
        """
        assert np.all(X.index == Y.index), "Y.index must be the same as X.index.  This is important because of custom cross-validation abilities.  Try something like Y = Y.loc[X.index,:]"
        if isinstance(Y, pd.Series):
            assert submodel is not None, "If `type(Y) is pd.Series` then `submodel` must be provided"
            Y = Y.to_frame(submodel)
        submodels_for_fitting = [submodel] if submodel is not None else self.submodels
        for submodel, y_subset in Y.loc[:,submodels_for_fitting].T.iterrows():
            # Get node data
            attributes = self.get_submodel_field(submodel=submodel, field="attributes")
            clf = self.get_submodel_field(submodel=submodel, field="clf")
            assert set(attributes) <= set(X.columns), f"Not all attributes from submodel={submodel} are in `X.columns`.\nSpecifically, {set(attributes) - set(X.columns)} are not in `X`"
            # Training data
            y_subset = y_subset.dropna()
            X_subset = X.loc[y_subset.index, attributes]
            # Fit classifier
            clf.fit(X_subset.values, y_subset)
            self.set_submodel_field(submodel=submodel, field="clf", obj=clf)
            # Copy training data
            restore_index_positions = False
            if self.graph.nodes[submodel]["cv"] is not None:
                if type(self.graph.nodes[submodel]["cv"]) != int:
                    restore_index_positions = True
            if all([restore_index_positions, ignore_submodel_index == False]):
                index = self.get_submodel_field(submodel=submodel, field="index")
                assert index is not None, f"`index` attribute of submodel={submodel} is None. Figure it out..."
                X_subset = X_subset.loc[index,:]
                y_subset = y_subset[index]
                assert y_subset.isnull().sum() == 0, f"indexing of `y` with `index` field provided by user yields `NaNs`.  One common cause is that you have provided a custom `cv` and `index` during `add_submodel` and now you are refitting with a different dataset or using manual cross-validation.  If you want to suppress the `index` dependency then use `ignore_submodel_index = True` but this has not been extensively checked and is not advised.  \n\nIf the above does not apply, please check these in particular:\n\n" + ','.join(y_subset[y_subset.isnull()].index.map(str))
#             # Inform the user that they are ignoring the submodel index
#             if all([restore_index_positions, ignore_submodel_index, self.verbose]):
#                 print(f"Ignoring the submodel `index` that was provided for custom cross-validation pairs in the `add_submodel` step.  Make sure this is what you want.  This is not advised and is better to repeat the `add_models` step without the `cv` and `index` assignments", file=sys.stderr)
            self.set_submodel_field(submodel=submodel, field="X", obj=X_subset.copy())
            self.set_submodel_field(submodel=submodel, field="y", obj=y_subset.copy())
            # Add to training samples set
            self.training_observations.update(set(y_subset.index))
        # Copy training data
        y_terminus = pd.Series(dict(map(lambda x:(x[0], Y.loc[x[0],x[1]]), Y.applymap(lambda x: x in self.paths).idxmax(axis=1).iteritems())))[Y.index]
        self.training_data["Y"] = Y.copy()
        self.training_data["y_terminus"] = y_terminus.copy()
        if self.copy_X:
            self.training_data["X"] = X.copy()
        return self


    # ==========
    # Predicting
    # ==========
    # Predict
    def predict(self, X, submodel=None,  mode = "directional"):
        if submodel is not None:
            clf = self.get_submodel_field(submodel=submodel, field="clf")
            attributes = self.get_submodel_field(submodel=submodel, field="attributes")
            y_hat = clf.predict(X.loc[:,attributes].values)
            return pd.Series(y_hat, index=X.index, name=submodel)
        else:
            available_modes = ["directional", "component", "diffuse"]
            assert mode in available_modes, f"`mode` argument must be in {available_modes}"
            if mode in ["component", "directional"]:
                data = list()
                for _submodel in self.submodels:
                    y_hat = self.predict(X, submodel=_submodel)
                    data.append(y_hat)
                df_predict =  pd.DataFrame(data).T
                df_predict.columns.name = "submodels"
                df_predict.index.name = "query"
                if mode == "component":
                    return df_predict
                else:
                    # Iterate through observations, go down the prediction paths, give the last prediction (this probably won't work if there are intermediate level predictions)
                    data_prediction_terminus = list()
                    for id_obsv, predictions in df_predict.iterrows():
                        node_current = predictions[0]
                        while node_current not in self.paths:
                            node_next = predictions[node_current]
                            node_current = node_next
                        node_terminus = node_current
                        data_prediction_terminus.append(node_terminus)
                    return pd.Series(data_prediction_terminus, index=df_predict.index, name="prediction")

            elif mode == "diffuse":
                df_diffuse = self.predict_proba(X=X, submodel=None, mode="diffuse")
                return df_diffuse.idxmax(axis=1)


    # Predict probabilities
    def predict_proba(self, X, submodel=None, mode="directional", logit_normalization="tss"):
        """
        """
        available_modes = ["directional", "diffuse"]
        assert mode in available_modes, f"mode must be in `{available_nodes}`"
        idx_obsvs = X.index
        if submodel is not None:
            clf = self.get_submodel_field(submodel=submodel, field="clf")
            attributes = self.get_submodel_field(submodel=submodel, field="attributes")
            Y_hat = clf.predict_proba(X.loc[:,attributes].values)
            return pd.DataFrame(Y_hat, index=idx_obsvs, columns=pd.Index(clf.classes_, name=submodel))
        else:
            data = list()
            for sm in self.submodels:
                clf = self.get_submodel_field(submodel=sm, field="clf")
                attributes = self.get_submodel_field(submodel=sm, field="attributes")
                Y_hat = clf.predict_proba(X.loc[:,attributes].values)
                df_submodel = pd.DataFrame(Y_hat,
                                           index=idx_obsvs,
                                           columns=pd.MultiIndex.from_tuples([*map(lambda x:(sm, x), clf.classes_)], names=["submodel", "classification"])
                )
                data.append(df_submodel)

            df_predict_proba =  pd.concat(data, axis=1)
            df_predict_proba.columns.name = "submodels"
            df_predict_proba.index.name = "query"

            self.add_subgraphs_from_probabilities(df_predict_proba)
            # Recommended method: Go in direction of highest probability in order
            if mode == "directional":
                return df_predict_proba
            # This will be an absolute path that looks for all of the potential paths
            if mode == "diffuse":
                if not hasattr(logit_normalization, "__call__"):
                    assert logit_normalization in ["tss", "softmax"], "`logit_normalization` must be either a callable to apply on axis=1 or a string in ['tss','softmax']"
                    logit_normalization = {"tss": self._tss, "softmax":self._softmax}[logit_normalization]
                d_obsv_target = defaultdict(dict)
                for id_obsv in idx_obsvs:
                    subgraph = self.subgraphs[id_obsv]
                    for target, path in self.paths.items():
                        subgraph_path = subgraph.subgraph(path[1:])
                        cumulative_score = np.product(np.array(list(map(lambda edge: subgraph_path.get_edge_data(edge[0], edge[1])["weight"], subgraph_path.edges()))))
                        d_obsv_target[id_obsv][target] = cumulative_score
                df_diffuse = pd.DataFrame(d_obsv_target).T.loc[idx_obsvs,:]
                return df_diffuse.apply(logit_normalization, axis=1)

    # Predict from a probability vector
    def predict_from_probas(self, probas:pd.Series):
        assert len(self.paths) > 0, "`paths` are empty.  Add the paths using `<model-name>.add_paths(paths)` with formatting analogous to `get_paths` from `Topology`"
        assert isinstance(probas, pd.Series), "`probas` should be a pd.Series"
        assert isinstance(probas.index, pd.MultiIndex), "`probas.index` should be a `pd.MultiIndex`"
        node_current = self.root_submodel
        while node_current not in self.paths:
            query_probas = probas[node_current]
            node_current = query_probas.idxmax()
        node_terminal = node_current
        return node_terminal

    # ================
    # Cross Validation
    # ================
    # Cross-validate submodels
    def cross_validate_submodels(self, X, Y, cv=None, n_jobs=1):
        """
        Cross-validate each model submodel individually to returns scores for each submodel in a pd.DataFrame

        If `cv` is None then it uses the `cv` from `add_submodel`
        `cv` can be overridden when `cv` arg in this method is not None
        """
        d_submodel_results = defaultdict(OrderedDict)
        for submodel in self.submodels:
            # Get node data
            attributes = self.get_submodel_field(submodel=submodel, field="attributes")
            clf = clone(self.get_submodel_field(submodel=submodel, field="clf"))

            if cv is None:
                assert self.get_submodel_field(submodel=submodel, field="cv") is not None, "`cv` must be provided in `add_submodel` if `cv is None` in this method"
                _cv = self.get_submodel_field(submodel=submodel, field="cv")
            else:
                _cv = cv
            index = self.get_submodel_field(submodel=submodel, field="index")
            self._check_cv(_cv, index)

            # If a custom cv is provided then there must be an index provided as well
            condition_1 = index is not None
            condition_2 = _cv is not None
            if all([condition_1, condition_2]):
                y_query = Y[submodel][index]
                X_query = X.loc[index,attributes]
            # If a integer cv is provided
            else:
                y_query = Y[submodel].dropna()
                X_query = X.loc[y_query.index,attributes]

            # Stats
            scores = model_selection.cross_val_score(clf, X=X_query.values, y=y_query.values, scoring="accuracy", cv=_cv, n_jobs=n_jobs)
            accuracy = np.mean(scores)
            sem = stats.sem(scores)
            # Update
            self.set_submodel_field(submodel=submodel, field="accuracy", obj=accuracy)
            self.set_submodel_field(submodel=submodel, field="sem", obj=sem)
            self.submodel_performance[submodel] = accuracy
            d_submodel_results[submodel]["accuracy"] = accuracy
            d_submodel_results[submodel]["sem"] = sem
            d_submodel_results[submodel]["scores"] = scores

        # Scores
        df_performance = pd.DataFrame(d_submodel_results).T.loc[:,["accuracy", "sem", "scores"]]
        df_performance.index.name = "submodel"
        self.add_weights_from_accuracy(df_performance["accuracy"])
        self._cross_validation_complete = True
        return df_performance

    # Cross validate entire model
    def cross_validate(self, X, Y, cv=None, mode="directional", labels=None):
        """
        Cross-validate the entire model
        """
        # Check input data
        assert set(X.index) == set(Y.index), "Y.index must have the same elements as X.index."
        index = X.index
        # Assign terminal predictions
        column_with_terminus = Y.applymap(lambda x: x in self.paths).idxmax(axis=1)
        y = pd.Series(map(lambda x:Y.loc[x, column_with_terminus[x]], Y.index), index=Y.index)
        if cv is None:
            submodel_root = list(set(map(lambda x: x[1], self.paths.values())))
            assert len(submodel_root) == 1, f"If cv is None, there must be a consensus level-1 submodel for all paths: {submodel_root}"
            condition_1 = self.get_submodel_field(submodel=submodel_root[0], field="cv") is not None
            condition_2 = self.get_submodel_field(submodel=submodel_root[0], field="index") is not None
            assert all([condition_1, condition_2]), f"To use the custom cross-validation provided in `add_submodel` for the root submodel `{submodel_root[0]}`, the user must provide both `cv` and `index` during submodel instantiation."
            cv = self.get_submodel_field(submodel=submodel_root[0], field="cv")
            index = self.get_submodel_field(submodel=submodel_root[0], field="index")
        # Organize data
        X = X.loc[index,:]
        Y = Y.loc[index,:]
        y = y[index]

        # Organize cross-validation pairs
        self._check_cv(cv, X.index)
        if self.verbose:
            cv_iterable = tqdm(enumerate(model_selection.check_cv(cv=cv, y=y, classifier=True).split(X=X, y=y)), "Cross-validating")
        else:
            cv_iterable = enumerate(model_selection.check_cv(cv=cv, y=y, classifier=True).split(X=X, y=y))

        # Cross-validation
        model_clone = copy.deepcopy(self)
        # Reset `cv` attribute on the clone
        [*map(lambda submodel: model_clone.set_submodel_field(submodel, "cv", None), self.submodels)]
        cv_results = list()
        for i, (idx_tr, idx_te) in cv_iterable:
            model_clone.fit(X=X.iloc[idx_tr,:], Y=Y.iloc[idx_tr,:])
            y_hat = model_clone.predict(X.iloc[idx_te,:], mode=mode)
            y_true = y[y_hat.index]
            score = np.mean(y_hat == y_true)
            cv_results.append(score)
        if labels is not None:
            assert len(labels) == len(cv_results), "Number of labels should be the same as the number of cross-validations"
            return pd.Series(cv_results, index=labels, name=mode)
        else:
            return np.asarray(cv_results)

    # ========
    # Plotting
    # ========
    def plot_prediction_paths(
            self,
            y_hat=None,
            pos=None,
            title=None,
            node_box_facecolor=None,
            node_fontsize = 15,
            edge_colors="#000000",
            show_weights=True,
            show_weight_labels=True,
            cumulative_weights=False,
            graphviz_style="dot",
            nodelabel_kws=dict(),
            # arrowsize=15,
            # arrowstyle="-|>",
            scale_height = 0.85,
            scale_width = 1.0,
            weight_func = lambda w, scale=0.1618: (w + scale) ** 10 + scale,
            error_scale_func = lambda w:w*256,
            error_linestyle="--",
            error_linewidth=1.0,
            alpha_error=0.05,
            title_kws=dict(),
            edge_kws=dict(),
            fig_kws=dict(),
            figsize=(8,8),
            style="seaborn-white",
            func_reduce=np.mean, # Changed 01 August 2018
            ax=None
        ):
        """
        https://matplotlib.org/2.0.1/examples/pylab_examples/fancyarrow_demo.html
        """

        # Make sure paths are specified
        assert bool(self.paths), "self.paths must be complete.  Please use `add_paths` method"

        # Defaults
        DEFAULT_edge_color = "#000000"
        _node_box_facecolor = {self.obsv_type:"#808080"}
        if node_box_facecolor is None:
            node_box_facecolor = dict()
        _node_box_facecolor.update(node_box_facecolor)
        _nodelabel_kws = {"node_box_facecolor":_node_box_facecolor, "font_size":node_fontsize}
        _nodelabel_kws.update(nodelabel_kws)
        _title_kws = dict(fontsize=18, fontweight="bold")
        _title_kws.update(title_kws)
        # _edge_kws = {"arrowsize":arrowsize, "arrowstyle":arrowstyle}
        _edge_kws = {}
        _edge_kws.update(edge_kws)
        _fig_kws = {"figsize":figsize}
        _fig_kws.update(fig_kws)


        # Get graph
        G_sem = None
        # Should method plot global graph structure with accuracies?
        if y_hat is None:
            G = self.graph
        else:
            # ==================
            # Single observation
            # ==================
            # Note the following operations are not the most computationally efficient but they are the most intuitive
            # If y_hat is a observation name convert it to subgraph
            if not is_dict_like(y_hat):
                if type(y_hat) in [str,tuple,int]:
                    assert y_hat in self.subgraphs, f"{y_hat} not in `self.subgraphs`.  Please add subgraph manually with `add_subgraphs_from_probabilities` or `predict_proba`"
                    G = self.subgraphs[y_hat]
            # If y_hat is a prediction vector convert it to subgraph
            if is_dict_like(y_hat):
                G = self._probabilities_to_subgraph(y_hat)
            # Multiple observations
            # =====================
            if not is_graph(y_hat):
                # If y_hat is a prediction dataframe convert it to an iterable of prediction vectors
                if isinstance(y_hat, pd.DataFrame):
                    y_hat = [*map(lambda x:x[1], y_hat.iterrows())]
                # If the object is an iterable
                condition_1 = type(y_hat) != tuple
                condition_2 = is_nonstring_iterable(y_hat)
                if  all([condition_1, condition_2]) :
                    edge_input = [*filter(lambda x: self.obsv_type in x, self.graph.edges())][0]
                    y_hat = list(y_hat)
                    assert len(y_hat) > 0, "`y_hat` is empty"
                    assert is_all_same_type(y_hat), "If `y_hat` is an iterable it must be all the same type"

                    # If y_hat is iterable of strings convert to iterable of subgraphs
                    if type(y_hat[0]) in [str,tuple,int]:
                        assert all(x in self.subgraphs for x in y_hat), "All items in `y_hat` must be in `self.subgraphs`"
                        y_hat = [*map(lambda x:self.subgraphs[x], y_hat)]

                    # If y_hat is iterable of subgraphs convert to iterable of prediction vectors
                    if is_graph(y_hat[0]):
                        y_hat = [*map(lambda x: self._subgraph_to_probabilities(x), y_hat)]

                    # If y_hat is an iterable of prediction vectors convert to dataframe and collapse
                    if is_dict_like(y_hat[0]):
                        # If objects are dictionaries, convert them to pd.Series
                        if is_dict(y_hat[0]): # This part may be weird...but just don't use dictionaries yet
                            y_hat = [*map(pd.Series, y_hat)]
                        Y_hat = pd.DataFrame(y_hat)
                        y_hat_sem = Y_hat.sem(axis=0)
                        y_hat_mean = Y_hat.apply(func_reduce, axis=0)
                        G = self._probabilities_to_subgraph(y_hat_mean)
                        G_sem = self._probabilities_to_subgraph(y_hat_sem)
                        if edge_input not in G.edges:
                            G.add_edge(*edge_input, weight=1.0)
                            G_sem.add_edge(*edge_input, weight=0.0)
                        else:
                            # Set weights for input connection
                            self.set_edge_field(edge_input, "weight", 1.0, G=G) # G[edge_input[0]][edge_input[1]]["weight"] = 1.0
                            self.set_edge_field(edge_input, "weight", 0.0, G=G_sem) # G_sem[edge_input[0]][edge_input[1]]["weight"] = 0.0
        # Convert to cumulative weights
        if cumulative_weights:
            edgelist = self.get_cumulative_weights_from_graph(G=G)
            G = nx.OrderedDiGraph(name=G.name)
            G.add_edges_from(edgelist)

        # If input subgraph does not have weights
        if show_weights or show_weight_labels:
            if not nx.get_edge_attributes(G, "weight"):
                show_weights = show_weight_labels = False
                if self.verbose:
                    print(f"Warning: No `weight` attribute for base graph because `self._cross_validation_complete = {self._cross_validation_complete}` setting the following `show_weights = show_weight_labels = False`", file=sys.stderr)

        # Positions for hierarchical topology
        if pos is None:
            G_scaffold = nx.OrderedDiGraph()
            G_scaffold.add_edges_from(self.graph.edges())
            pos = nx.nx_agraph.graphviz_layout(G_scaffold, prog=graphviz_style, root=self.obsv_type)

        if edge_colors is None:
            edge_colors = DEFAULT_edge_color

        # Weights
        if weight_func is None:
            weight_func = lambda w:w
        if error_scale_func is None:
            error_scale_func = lambda w:w
        edge_labels = dict(map(lambda edge_data: (edge_data[0], "{0:.3f}".format(edge_data[1])), nx.get_edge_attributes(G, "weight").items()))
        if show_weights:
            weights = np.array([G[u][v]['weight']for u,v in G.edges()])
            if G_sem is not None:
                weights_sem = np.array([G_sem[u][v]['weight']for u,v in G_sem.edges()])
        else:
            weights = np.asarray([1.0]*G.number_of_edges())
            if G_sem is not None:
                weights_sem = np.asarray([1.0]*G.number_of_edges())


        # Plotting graph
        # ==============
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(**_fig_kws)
            else:
                fig = plt.gcf()
            # Weights
            # -------
            # Draw sem error
            if G_sem is not None:
                # Acknowledgments to @importanceofbeingernest for plotting outline
                # https://stackoverflow.com/questions/55911075/how-to-plot-the-outline-of-the-outer-edges-on-a-matplotlib-line-in-python/56030879#56030879
                # OutlineCollection is not working with fancy arrow patches
                error_line_collection = nx.draw_networkx_edges(G_sem, pos, alpha=alpha_error, edges=G_sem.edges(), edge_color=edge_colors, width=error_scale_func(weights_sem), ax=ax, arrows=False, **_edge_kws)
                for path, lw in zip(error_line_collection.get_paths(), error_line_collection.get_linewidths()):
                    error_outline_collection = OutlineCollection(error_line_collection, ax=ax, linewidth=error_linewidth,  alpha=alpha_error, linestyle=error_linestyle, edgecolor=edge_colors, facecolor="none")

            # Draw accuracies
            nx.draw_networkx_edges(G, pos, edges=G.edges(), edge_color=edge_colors, width=weight_func(weights), ax=ax,**_edge_kws )
            if show_weight_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, ax=ax)
            # Nodes
            # -----
            draw_networkx_labels_with_box(G, pos, ax=ax, **_nodelabel_kws)

            # # Weights
            # # -------
            # # Draw sem halo
            # if G_sem is not None:
            #     nx.draw_networkx_edges(G_sem, pos, alpha=alpha_error, edges=G_sem.edges(), arrowstyle="-", edge_color=edge_colors, width=error_scale_func(weights_sem), ax=ax, **{k:_edge_kws[k]  for k in _edge_kws if k != "arrowstyle"})
            # # Draw accuracies
            # nx.draw_networkx_edges(G, pos, edges=G.edges(), edge_color=edge_colors, width=weight_func(weights), ax=ax,**_edge_kws )
            # if show_weight_labels:
            #     nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, ax=ax)
            # # Nodes
            # # -----
            # draw_networkx_labels_with_box(G, pos, ax=ax, **_nodelabel_kws)
            # Scaling
            # -------
            A_pos = np.stack(pos.values())
            width = A_pos[:,0].max() - A_pos[:,0].min()
            width = np.sqrt(width) #- np.log(width)
            height = A_pos[:,1].max() - A_pos[:,1].min()
            height = (np.sqrt(height) - np.log(height))
            fig.set_figwidth(width*scale_width)
            fig.set_figheight(height*scale_height)
            # Labels
            if title is None:
                title = G.name
            if title is not None:
                ax.set_title(title, **_title_kws)
            # Aesthetics
            # ax.autoscale(True)
            ax.grid(False)
            ax.axis('off')

            return fig, ax


    # Plotting
    def plot_model(self,  cumulative_weights=False, pos=None, title=None, node_box_facecolor=None, show_weights=True, graphviz_style="dot", edge_kws=dict(), arrowsize=15, arrowstyle="wedge", scale_height = 0.85, scale_width = 1.0, weight_func = lambda w, scale=0.1618: (w + scale) ** 10 + scale, title_kws=dict(), style="seaborn-white", nodelabel_kws=dict(), ax=None):
        """
        Recommended `graphviz_style` options: ["neato", "dot", "sfdp"]
        """
        fig, ax = self.plot_prediction_paths(y_hat=None,  cumulative_weights=cumulative_weights, pos=pos, title=title, node_box_facecolor=node_box_facecolor,  show_weights=show_weights, show_weight_labels=show_weights, graphviz_style=graphviz_style, edge_kws=edge_kws, arrowsize=arrowsize, arrowstyle=arrowstyle, scale_height =scale_height, scale_width = scale_width, weight_func = weight_func, title_kws=title_kws, style=style, nodelabel_kws=nodelabel_kws, ax=ax)
        return fig, ax
