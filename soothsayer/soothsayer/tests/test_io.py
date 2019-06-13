# Imports
import sys, os, pathlib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from collections import *
import soothsayer as sy
# ========================
# Current working directory
# ========================
as_absolute = lambda path: os.path.abspath(os.path.join(__file__, "../", path))

# ========================
# Generating test datasets
# ========================
# # X
# iris = load_iris()
# X_iris = pd.DataFrame(
#     data=iris.data,
#     index=[*map(lambda i:f"iris_{i}",range(iris.data.shape[0]))],
#     columns=[*map(lambda x: x.split(" (cm)")[0].replace(" ","_"), iris.feature_names)]
# )
#
# # y
# y_iris = pd.Series(iris.target, index=X_iris.index, name="species").map(lambda i:iris.target_names[i])
X_iris, y_iris = sy.utils.get_iris_data(return_data=["X","y"])

# ========================
# Save data
# ========================
def test_write_table():
    #tsv
    sy.write_dataframe(X_iris, as_absolute("./data/X_iris.tsv"))
    sy.write_dataframe(y_iris.to_frame(), as_absolute("./data/y_iris.labels.tsv.gz"))
    # gzipped tsv
    sy.write_dataframe(X_iris, as_absolute("./data/X_iris.tsv.gz"))
    # bz2 pickle
    sy.write_dataframe(X_iris, as_absolute("./data/X_iris.pbz2"))
    # gzipped pickle
    sy.write_dataframe(X_iris, as_absolute("./data/X_iris.pgz"))
    # pickle
    sy.write_dataframe(X_iris, as_absolute("./data/X_iris.pkl"))
    # excel
    sy.write_dataframe(OrderedDict([("X_iris", X_iris), ("y_iris", y_iris.to_frame())]), as_absolute("./data/iris.xlsx"))

def test_read_csv():
    filepaths = [
    "./data/X_iris.tsv",
    "./data/y_iris.labels.tsv.gz",
    "./data/X_iris.tsv.gz",
    "./data/X_iris.pbz2",
    "./data/X_iris.pgz",
    "./data/X_iris.pkl",
    "./data/iris.xlsx"
    ]
    for path in filepaths:
        sy.read_dataframe(as_absolute(path))
