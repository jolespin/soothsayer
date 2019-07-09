
# Denoising Iris dataset + 96 N(0,1) attributes using *Clairvoyance*

The following tutorial uses the Iris dataset (n=150 samples, m=4 attributes, c=3 classes) with 96 N(0,1) distributed noise attributes.  Illustrates how to use *Clairvoyance* and how to interpret the output files.


```python
# Import
import os,sys
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soothsayer as sy
from sklearn.model_selection import StratifiedKFold

%matplotlib inline
```

## Dataset


```python
# ==============
# Create dataset
# ==============
# Load in Iris dataset and add noise
X_iris, y_iris = sy.utils.get_iris_data(return_data=["X","y"], noise=96) #96 + 4 = 100 attributes

# Noise is N(0,1) so let's zscore normalize the actual iris data to keep the scales similar
X_iris.iloc[:,:4] = sy.transmute.normalize(X_iris.iloc[:,:4], method="zscore", axis=0)

# Get colors
chromatic_iris = sy.Chromatic.from_classes(y_iris, palette="Set2")

# Custom cross-validation pairs (not required)
K = 10
skf = StratifiedKFold(n_splits=K)
cv_data = defaultdict(dict)
for i, (idx_tr, idx_te) in enumerate(skf.split(X_iris, y_iris), start=1):
    label = "cv={}".format(i)
    cv_data[label]["Training"] = list(idx_tr)
    cv_data[label]["Testing"] = list(idx_te)
df_cv = pd.DataFrame(cv_data).T.loc[:,["Training", "Testing"]]

# Save the data

sy.io.write_dataframe(X_iris, "../Data/Iris/X_iris.noise_96.zscore.pbz2", create_directory_if_necessary=True) # This doesn't have to be a bz2 pickle object.  Just showing that it can work
sy.io.write_dataframe(y_iris.to_frame(), "../Data/Iris/y_iris.species.tsv.gz")
sy.io.write_dataframe(df_cv, "../Data/Iris/cv-with-labels.tsv")
```

## Visualize


```python
# =========================================
# Original Iris dataset (Zscore normalized)
# =========================================


# Symmetric pairwise distance matrix
df_dism = sy.symmetry.pairwise(X_iris.loc[:,['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], metric="euclidean", axis=0)

# Hierarchical clustering
ach_iris = sy.Agglomerative(df_dism,  name="iris")
ach_iris.add_secondary_class(name="species", mapping=y_iris, class_colors=chromatic_iris.class_colors)
ach_iris.plot()

# Ordination
with plt.style.context("seaborn-white"):
    fig, axes = plt.subplots(figsize=(13,5), ncols=2)
    # Principal Coordinates Analysis
    pcoa_iris = sy.ordination.PrincipalCoordinatesAnalysis(df_dism)
    pcoa_iris.plot(c=chromatic_iris.obsv_colors,  title="Principal Coordinates Analysis", ax=axes[0])

    # t_SNE
    tsne_iris = sy.ordination.Manifold(df_dism)
    tsne_iris.plot(c=chromatic_iris.obsv_colors, legend=chromatic_iris.class_colors, title="$t$-Distributed Schocastic Neighborhood Embeddings", ax=axes[1])
```

    Inferred mode as `dissimilarity`



![png](output_5_1.png)



![png](output_5_2.png)



```python
# =========================
# After adding N(0,1) noise
# =========================

# Symmetric pairwise distance matrix
df_dism = sy.symmetry.pairwise(X_iris, metric="euclidean", axis=0)

# Hierarchical clustering
ach_iris = sy.Agglomerative(df_dism, name="iris")
ach_iris.add_secondary_class(name="species", mapping=y_iris, class_colors=chromatic_iris.class_colors)
ach_iris.plot()

# Ordination
with plt.style.context("seaborn-white"):
    fig, axes = plt.subplots(figsize=(13,5), ncols=2)
    # Principal Coordinates Analysis
    pcoa_iris = sy.ordination.PrincipalCoordinatesAnalysis(df_dism)
    pcoa_iris.plot(c=chromatic_iris.obsv_colors,  title="Principal Coordinates Analysis", ax=axes[0])

    # t_SNE
    tsne_iris = sy.ordination.Manifold(df_dism)
    tsne_iris.plot(c=chromatic_iris.obsv_colors, legend=chromatic_iris.class_colors, title="$t$-Distributed Schocastic Neighborhood Embeddings", ax=axes[1])
```

    Inferred mode as `dissimilarity`



![png](output_6_1.png)



![png](output_6_2.png)


## Documentation for Clairvoyance

```bash
# run_soothsayer.py clairvoyance -h
usage: 
    soothsayer:clairvoyance v0.3_2018-08-28
     [-h]
                                                         [-X ATTRIBUTE_MATRIX]
                                                         [-y TARGET_VECTOR]
                                                         [-e ENCODING]
                                                         [--normalize NORMALIZE]
                                                         [-m MODEL_TYPE]
                                                         [--n_iter N_ITER]
                                                         [--random_state RANDOM_STATE]
                                                         [--n_jobs N_JOBS]
                                                         [--min_threshold MIN_THRESHOLD]
                                                         [--random_mode RANDOM_MODE]
                                                         [--percentiles PERCENTILES]
                                                         [-n NAME]
                                                         [-o OUT_DIR]
                                                         [--attr_type ATTR_TYPE]
                                                         [--class_type CLASS_TYPE]
                                                         [--method METHOD]
                                                         [--adaptive_range ADAPTIVE_RANGE]
                                                         [--adaptive_steps ADAPTIVE_STEPS]
                                                         [--cv CV]
                                                         [--min_bruteforce MIN_BRUTEFORCE]
                                                         [--early_stopping EARLY_STOPPING]
                                                         [--compression COMPRESSION]
                                                         [--pickled PICKLED]
                                                         [--save_kernel SAVE_KERNEL]
                                                         [--save_model SAVE_MODEL]
                                                         [--save_data SAVE_DATA]

optional arguments:
  -h, --help            show this help message and exit
  -X ATTRIBUTE_MATRIX, --attribute_matrix ATTRIBUTE_MATRIX
                        Input: Path/to/Tab-separated-value.tsv of attribute
                        matrix (rows=samples, cols=attributes)
  -y TARGET_VECTOR, --target_vector TARGET_VECTOR
                        Input: Path/to/Tab-separated-value.tsv of target
                        vector (rows=samples, column=integer target)
  -e ENCODING, --encoding ENCODING
                        Input: Path/to/Tab-separated-value.tsv of encoding.
                        Column 1 has enocded integer and Column 2 has string
                        representation. No header! [Default: None]
  --normalize NORMALIZE
                        Normalize the attribute matrix. Valid Arguments:
                        {'tss', 'zscore', 'quantile', None}: Warning, apply
                        with intelligence. For example, don't use total-sum-
                        scaling for fold-change values. [Experimental:
                        2018-May-03]
  -m MODEL_TYPE, --model_type MODEL_TYPE
                        Model types in ['logistic','tree'] represented by a
                        comma-separated list [Default: 'logistic,tree']
  --n_iter N_ITER       Number of iterations to split data and fit models.
                        [Default: 10]
  --random_state RANDOM_STATE
                        [Default: 0]
  --n_jobs N_JOBS       [Default: 1]
  --min_threshold MIN_THRESHOLD
                        Minimum accuracy for models when considering
                        coeficients for weights. Can be a single float or list
                        of floats separated by `,` ranging from 0-1. If None
                        it takes an aggregate weight. [Default: 0.0,None]
  --random_mode RANDOM_MODE
                        Random mode 0,1, or 2. 0 uses the same random_state
                        each iteration. 1 uses a different random_state for
                        each parameter set. 2 uses a different random_state
                        each time. [Default: 0]
  --percentiles PERCENTILES
                        Iterative mode takes in a comma-separated list of
                        floats from 0-1 that dictate how many of the
                        attributes will be used [Default:
                        0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99]
  -n NAME, --name NAME  Name for data [Default: Date]
  -o OUT_DIR, --out_dir OUT_DIR
                        Output: Path/to/existing-directory-for-output. Do not
                        use `~` to specify /home/[user]/ [Default: cwd]
  --attr_type ATTR_TYPE
                        Name of attribute type (e.g. gene, orf, otu, etc.)
                        [Default: 'attr']
  --class_type CLASS_TYPE
                        Name of class type (e.g. phenotype, MOA, etc.)
                        [Default: 'class']
  --method METHOD       {'adaptive', 'bruteforce'} [Default: 'bruteforce']
  --adaptive_range ADAPTIVE_RANGE
                        Adaptive range in the form of a string of floats
                        separated by commas [Default:'0.0,0.5,1.0']
  --adaptive_steps ADAPTIVE_STEPS
                        Adaptive stepsize in the form of a string of ints
                        separated by commas. len(adaptive_range) - 1
                        [Default:'1,10']
  --cv CV               Number of steps for percentiles when cross validating
                        and dropping attributes. If cv=0 then it skips cross-
                        validation step. [Default: 5]
  --min_bruteforce MIN_BRUTEFORCE
                        Minimum number of attributes to adjust to bruteforce.
                        If value is 0 then it will not change method [Default:
                        150]
  --early_stopping EARLY_STOPPING
                        Stopping the algorthm if a certain number of
                        iterations do not increase the accuracy. Use -1 if you
                        don't want to use `early_stopping` [Default: 100]
  --compression COMPRESSION
  --pickled PICKLED
  --save_kernel SAVE_KERNEL
                        Save the kernel [Default: True]
  --save_model SAVE_MODEL
                        Save the model [Default: False]
  --save_data SAVE_DATA
                        Save the model [Default: True]
```


```bash
%%bash
# ====================
# Running Clairvoyance
# ====================
X="../Data/Iris/X_iris.noise_96.zscore.pbz2"
y="../Data/Iris/y_iris.species.tsv.gz"
cv="../Data/Iris/cv-with-labels.tsv"
output_directory="../Data/Iris/Clairvoyance_Output"
mkdirs -p ${output_directory}
# Run algorithm
time(run_soothsayer.py clairvoyance -X ${X} -y ${y} -n iris -o ${output_directory} --cv ${cv} > ${output_directory}/clairvoyance.o)
```

    bash: line 8: mkdirs: command not found
    = == === ===== ======= ============ =====================
    logistic
    . .. ... ..... ....... ............ .....................
    logistic | Percentile=0.0 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  3.45it/s]
    logistic | Percentile=0.0 | X.shape = (150, 100) | Baseline score(../Data/Iris/X_iris.noise_96.zscore.pbz2) = 0.86667
    logistic | CV | Percentile=0.0 | Minimum threshold=0.0: 100%|██████████| 99/99 [00:02<00:00, 46.63it/s]
    logistic | CV | Percentile=0.0 | Minimum threshold=None: 100%|██████████| 99/99 [00:02<00:00, 45.33it/s]
    logistic | Percentile=0.3 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  3.86it/s]
    logistic | Percentile=0.3 | X.shape = (150, 70) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.3/X.subset.pbz2) = 0.86667
    logistic | CV | Percentile=0.3 | Minimum threshold=0.0: 100%|██████████| 69/69 [00:01<00:00, 53.32it/s]
    logistic | CV | Percentile=0.3 | Minimum threshold=None: 100%|██████████| 69/69 [00:01<00:00, 46.44it/s]
    logistic | Percentile=0.5 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  3.95it/s]
    logistic | Percentile=0.5 | X.shape = (150, 50) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.5/X.subset.pbz2) = 0.86667
    logistic | CV | Percentile=0.5 | Minimum threshold=0.0: 100%|██████████| 49/49 [00:00<00:00, 52.22it/s]
    logistic | CV | Percentile=0.5 | Minimum threshold=None: 100%|██████████| 49/49 [00:00<00:00, 52.96it/s]
    logistic | Percentile=0.75 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  4.78it/s]
    logistic | Percentile=0.75 | X.shape = (150, 25) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.75/X.subset.pbz2) = 0.87333
    logistic | CV | Percentile=0.75 | Minimum threshold=0.0: 100%|██████████| 24/24 [00:00<00:00, 57.20it/s]
    logistic | CV | Percentile=0.75 | Minimum threshold=None: 100%|██████████| 24/24 [00:00<00:00, 51.54it/s]
    logistic | Percentile=0.9 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  5.05it/s]
    logistic | Percentile=0.9 | X.shape = (150, 10) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.9/X.subset.pbz2) = 0.94
    logistic | CV | Percentile=0.9 | Minimum threshold=0.0: 100%|██████████| 9/9 [00:00<00:00, 44.01it/s]
    logistic | CV | Percentile=0.9 | Minimum threshold=None: 100%|██████████| 9/9 [00:00<00:00, 36.25it/s]
    logistic | Percentile=0.95 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.30it/s]
    logistic | Percentile=0.95 | X.shape = (150, 5) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.95/X.subset.pbz2) = 0.93333
    logistic | CV | Percentile=0.95 | Minimum threshold=0.0: 100%|██████████| 4/4 [00:00<00:00, 42.16it/s]
    logistic | Percentile=0.96 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.18it/s]
    logistic | Percentile=0.96 | X.shape = (150, 4) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.96/X.subset.pbz2) = 0.93333
    logistic | CV | Percentile=0.96 | Minimum threshold=0.0: 100%|██████████| 3/3 [00:00<00:00, 40.33it/s]
    logistic | Percentile=0.97 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.32it/s]
    logistic | Percentile=0.97 | X.shape = (150, 3) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.97/X.subset.pbz2) = 0.94
    logistic | CV | Percentile=0.97 | Minimum threshold=0.0: 100%|██████████| 2/2 [00:00<00:00, 37.93it/s]
    logistic | Percentile=0.98 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.45it/s]
    logistic | Percentile=0.98 | X.shape = (150, 2) | Baseline score(../Data/Iris/Clairvoyance_Output/logistic/percentile_0.98/X.subset.pbz2) = 0.96
    logistic | CV | Percentile=0.98 | Minimum threshold=0.0: 100%|██████████| 1/1 [00:00<00:00, 28.67it/s]
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    Skipping current iteration 0.99 because there is/are only 1 attribute[s]
    . .. ... ..... ....... ............ .....................
    logistic | Percentile=0.9 | Minimum threshold = 0.0 | Baseline score = 0.86667 | Best score = 0.96 | ∆ = 0.093333 | n_attrs = 2
    . .. ... ..... ....... ............ .....................
    
    = == === ===== ======= ============ =====================
    tree
    . .. ... ..... ....... ............ .....................
    tree | Percentile=0.0 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  4.18it/s]
    tree | Percentile=0.0 | X.shape = (150, 100) | Baseline score(../Data/Iris/X_iris.noise_96.zscore.pbz2) = 0.95333
    tree | CV | Percentile=0.0 | Minimum threshold=0.0: 100%|██████████| 99/99 [00:02<00:00, 44.47it/s]
    tree | Percentile=0.3 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  4.68it/s]
    tree | Percentile=0.3 | X.shape = (150, 70) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.3/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.3 | Minimum threshold=0.0: 100%|██████████| 69/69 [00:01<00:00, 42.81it/s]
    tree | Percentile=0.5 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:02<00:00,  4.87it/s]
    tree | Percentile=0.5 | X.shape = (150, 50) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.5/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.5 | Minimum threshold=0.0: 100%|██████████| 49/49 [00:00<00:00, 59.04it/s]
    tree | Percentile=0.75 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.31it/s]
    tree | Percentile=0.75 | X.shape = (150, 25) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.75/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.75 | Minimum threshold=0.0: 100%|██████████| 24/24 [00:00<00:00, 66.97it/s]
    tree | Percentile=0.9 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.50it/s]
    tree | Percentile=0.9 | X.shape = (150, 10) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.9/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.9 | Minimum threshold=0.0: 100%|██████████| 9/9 [00:00<00:00, 66.90it/s]
    tree | Percentile=0.95 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.77it/s]
    tree | Percentile=0.95 | X.shape = (150, 5) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.95/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.95 | Minimum threshold=0.0: 100%|██████████| 4/4 [00:00<00:00, 63.72it/s]
    tree | Percentile=0.96 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.81it/s]
    tree | Percentile=0.96 | X.shape = (150, 4) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.96/X.subset.pbz2) = 0.96667
    tree | CV | Percentile=0.96 | Minimum threshold=0.0: 100%|██████████| 3/3 [00:00<00:00, 60.94it/s]
    tree | Percentile=0.97 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.82it/s]
    tree | Percentile=0.97 | X.shape = (150, 3) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.97/X.subset.pbz2) = 0.96667
    tree | CV | Percentile=0.97 | Minimum threshold=0.0: 100%|██████████| 2/2 [00:00<00:00, 57.15it/s]
    tree | Percentile=0.98 | Permuting samples and fitting models: 100%|██████████| 10/10 [00:01<00:00,  5.75it/s]
    tree | Percentile=0.98 | X.shape = (150, 2) | Baseline score(../Data/Iris/Clairvoyance_Output/tree/percentile_0.98/X.subset.pbz2) = 0.95333
    tree | CV | Percentile=0.98 | Minimum threshold=0.0: 100%|██████████| 1/1 [00:00<00:00, 45.07it/s]
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    /Users/jespinoz/anaconda/envs/µ_env/lib/python3.6/site-packages/pandas/plotting/_core.py:1001: UserWarning: Attempting to set identical left==right results
    in singular transformations; automatically expanding.
    left=1.0, right=1.0
      ax.set_xlim(left, right)
    Skipping current iteration 0.99 because there is/are only 1 attribute[s]
    . .. ... ..... ....... ............ .....................
    tree | Percentile=0.96 | Minimum threshold = 0.0 | Baseline score = 0.95333 | Best score = 0.96667 | ∆ = 0.013333 | n_attrs = 3
    . .. ... ..... ....... ............ .....................
    
    
    real	3m28.155s
    user	3m36.305s
    sys	0m2.698s


## *Clairvoyance* Output


```python
# Load in the synopsis (there are more intermediate files as well)
df_synopsis = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/iris__synopsis.tsv")
# Check the accuracy for Logistic Regression and Decision Tree-based models
df_synopsis.groupby("model_type")["accuracy"].max()
```




    model_type
    logistic    0.960000
    tree        0.966667
    Name: accuracy, dtype: float64




```python
# Check the accuracy of the models
for model_type, df in df_synopsis.groupby("model_type"):
    df = df.sort_values(["accuracy", "sem"], ascending=[False, True])
    print(sy.utils.format_header(model_type, "-"), df.iloc[0,:].drop(["baseline_score_of_current_percentile", "random_state", "delta"]), sep="\n")
```

    --------
    logistic
    --------
    model_type                                  logistic
    hyperparameters          {'C': 1.0, 'penalty': 'l1'}
    percentile                                       0.9
    min_threshold                                      0
    baseline_score                              0.866667
    accuracy                                        0.96
    sem                                        0.0147406
    num_attr_included                                  2
    attr_set             ['petal_width', 'petal_length']
    Name: 8, dtype: object
    ----
    tree
    ----
    model_type                                                                         tree
    hyperparameters      {'criterion': 'gini', 'min_samples_leaf': 3, 'max_features': None}
    percentile                                                                         0.96
    min_threshold                                                                         0
    baseline_score                                                                 0.953333
    accuracy                                                                       0.966667
    sem                                                                           0.0111111
    num_attr_included                                                                     3
    attr_set                                ['petal_width', 'petal_length', 'sepal_length']
    Name: 20, dtype: object



```python
# =======================================
# Attribute weights (model_type=logisitc)
# =======================================

# Class agnostic weights
# ----------------------

W_logistic = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/logistic__pctl_0.9__t_0.0__weights.tsv.gz")

with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(W_logistic, vmin=0, vmax=1, cmap=plt.cm.magma_r, annot=True, cbar_kws={"label":"W"}, ax=ax, edgecolor="white", linewidth=1)
    ax.set_title("model_type=logistic", fontsize=15, fontweight="bold")
    ax.set_ylabel(None)

# Check that all weights sum to 1
assert np.allclose(W_logistic.sum(axis=0), 1)

# Class-specific weights
# ----------------------
W_logistic = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/logistic__pctl_0.9__t_0.0__weights.class-specific.tsv.gz")

with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(W_logistic, vmin=0, vmax=1, cmap=plt.cm.magma_r, annot=True, cbar_kws={"label":"W"}, ax=ax, edgecolor="white", linewidth=1)
    ax.set_title("model_type=logistic", fontsize=15, fontweight="bold")
    ax.set_ylabel(None)

# Check that all weights sum to 1
assert np.allclose(W_logistic.sum(axis=0), 1.0)
```


![png](output_12_0.png)



![png](output_12_1.png)



```python
# =======================================
# Attribute weights (model_type=tree)
# =======================================

# Class agnostic weights
# ----------------------

W_tree = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/tree__pctl_0.96__t_0.0__weights.tsv.gz")

with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots(figsize=(5,3))
    sns.heatmap(W_tree, vmin=0, vmax=1, cmap=plt.cm.magma_r, annot=True, cbar_kws={"label":"W"}, ax=ax, edgecolor="white", linewidth=1)
    ax.set_title("model_type=tree", fontsize=15, fontweight="bold")
    ax.set_ylabel(None)

# Check that all weights sum to 1
assert np.allclose(W_tree.sum(axis=0), 1)

# # Class-specific weights
# # ----------------------
# # In future versions
```


![png](output_13_0.png)



```python
# Let's check the distribution of weights from the different thresholds at the "best" percentile
W_tree__distribution = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/tree__pctl_0.96__weights.tsv.gz")
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots(figsize=(5,3))
    W_tree__distribution.plot(kind="kde", ax=ax)
    ax.set_title("tree__pctl_0.96", fontsize=15, fontweight="bold")
    ax.legend(**{ 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5), "fontsize":12})
    ax.xaxis.grid(True)


W_logistic__distribution = sy.io.read_dataframe("../Data/Iris/Clairvoyance_Output/iris__synopsis/logistic__pctl_0.9__weights.tsv.gz")
with plt.style.context("seaborn-white"):
    fig, ax = plt.subplots(figsize=(5,3))
    W_logistic__distribution.plot(kind="kde", ax=ax)
    ax.set_title("logistic__pctl_0.9", fontsize=15, fontweight="bold")
    ax.legend(**{ 'frameon': True, 'facecolor': 'white', 'edgecolor': 'black', 'loc': 'center left', 'bbox_to_anchor': (1, 0.5), "fontsize":12})
    ax.xaxis.grid(True)


```


![png](output_14_0.png)



![png](output_14_1.png)


## Summary
The 96 noise features obscures the classes and creates an opportunity to illustrate the usage of *Clairvoyance*. 

From these parameters, we can see that Decision Trees slightly outperform Logistic Regression; though, the results are very similar.  

Despite the model type, we see the `petal_width` and `petal_length` are the most determinant features when discriminating the species.  
