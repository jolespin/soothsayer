# ===========
# DEVELOPMENT
# ===========
Once I have more time and create soothsayer v1.0 then I will make all of this formatted consistently

# ========
# Changes:
# ========
* 2023.12.22 - Fixed critical error in `Agglomerative` that gave incorrect leaf names.  It's because the disismilarity matrix was converted to a `Symmetric` object but the node labels changed order.
* 2023.8.22 - Changed `condensed_to_dense` to `condensed_to_redundant` and `dense_to_condensed` to `redundant_to_condensed`.  Also changed `cluster_homogeneity` to `edge_cluster_cooccurrence`
* 2022.4.4 - Fixed the edges - edgelist in networkx draw_networkx_edges function
* 2022.2.14 - Added `protest` method to `Procrustes` and cleaned up plotting function. Fixed ha and horzintalalignment double call in Agglomerative
* 2022.02.06 - Added `Procrustes` to ordination and replaced .as_matrix() calls with .values
* 2022.01.07 - Added `get_mixed_feature_pipeline_estimator` and added axis argument to `pd_prepend_level_to_index`
* 2021.07.27 - Added `plot_multiindexed_heatmap` and `iterate_nested_dict`
* 2021.05.19 - Added `intersection` and `union` functions from `soothsayer_utils`.  Also added 2 wrappers around scipy.stats.entropy (`shannon_entropy` and `kullback_leibler_divergence`) and a wrapper around scipy.stats.ttest_ind (`welchs_ttest`)
* 2021.04.27 - Reimplemented `normalize_css` in Python to remove dependency on metagenomeSeq.   Note, `metagenomeSeq` is confusing because it uses `p` for quantile which sounds like it's referring to `percentiles`.  To make it more consistent with `numpy` and `scipy` I switched `p` to `q`.  Also, I removed the `propr` dependency because it's not used at all now that I've made the `compositional` Python package.
* 2021.04.12 - Changed `cluster_modularity` to `cluster_homogeneity` in EnsembleNetworkX and also added `community_detection`
* 2021.03.22 - Deprecated some KEGG databases because versions couldn't be confirmed.  Added a read_kegg_json in soothsayer.io.  Fixed an error with the MANIFEST.in not including sub-directories in db and non-.pbz2 files such as the .flat and .json files
* 2021.03.05 - Added edgeR's exactTest, edgeR's glmLRT, and fixed np.all incompatibilities (https://github.com/pandas-dev/pandas/issues/40259)
* 2021.01.21 - Added `markerscale=2` to LEGEND_KWS
* 2021.01.13 - FINALLY removed `WGCNA` dependence.  I have recoded all of the necessary functions in Python.  This will make installation MUCH easier.  New functions include: density, heterogeneity, centralization, scalefree_topology_estimation, and an updated determine_soft_threshold.  I've been trying to do this for years...but never fully committed.
* 2021.01.06 - Added `differential_abundance` to `statistics` using `ALDEx2` in `r_wrappers`
* 2020.12.16 - Fixed a ddof=0 and ddof=1 incompatibility.  Now all ddof=1 in compositional package.
* 2020.09.17 - Fixed LinAlgError in plot_compositional from seaborn update to v0.11
* 2020.07.21 - Reimplemented PrincipalCoordinatesAnalysis and PrincipalComponentsAnalysis classes (and base classes).  They now have the ability to plot biplots, get loadings, and get contribution of feature classes to each PC.
* 2020.07.21 - Added `add_objects_to_globals` function to add objects from accessory packages like hive_networkx, ensemble_networkx, soothsayer_utils, and compositional
* 2020.07.01 - Created EnsembleAssociationNetwork class
* 2020.06.15-30 Make hive_networkx its own package.  Moved Symmetric and Hive (along with other functions)
* 2020.05.04 - Adapted code to support py3.7/8 and rpy2 v3.x.  I only ran a few tests so hopefullyt his doesn't break
* 2020.05.20 - Fixed nx.add_path deprecation for Hierarchical Classifier
* 2020.05.20 - Created "infer_tree_type", "check_polytomy", "prune_tree", and "is_leaf" in utils. 
* 2020.05.20 - Changed "name_ete_nodes to "name_tree_nodes" (made skbio compatible) and moved to utils
* 2020.05.20 - Made dependency on `compositional`.  Replaced transform functions in `soothsayer.transmute.normalization` and added functions in `soothsayer.symmetry`.
* 2020.05.20 - Removed automatic maximum threads for WGCNA
* 2020.06.01 - Removed dependencies on R packages but rpy2 is still necessary
* 2020.06.04 - Added "component_type" to `alpha_diversity`



# Bugs:
# =====
* 2019.12.06 For some reason adjusting something in regression broke the ability to do sy.r_wrappers.philr.normalize_philr  but you can still do from soothsayer.r_wrappers import philr

# ===========
# __future__:
# ===========

# Clairvoyance
# ------------
* Remove name in output files [Fixed v2022.01.06]
* Add weights in the synopsis file [Fixed v2022.01.06]
* Remove feature type from synopsis names [Fixed v2022.01.06]
* Change `attributes` to `features` [Fixed v2022.01.06]
* Write synopsis file after each model_type is finished [Fixed v2022.01.06]
* Output only the best models [Fixed v2022.01.06]
* For normalization, normalize based on the original data not the data for each iteration. For example, this wouldn't make sense for CLR.  [Fixed v2022.01.06]
* Remove all instances of --encoding
* Make --save_kernel, save_model, etc. flags
* Incorporate elasticnet for LogisticRegression

# Soothsayer
# ----------
* Remove signed from pairwise and clean up function
* Make Agglomerative its own package and use skbio instead of Ete3
* Make HierarchicalClassifier its own package and use skbio instead of Ete3
* Change `create_pca_model` to something else...
* Collapse topology so it uses Agglomerative instead
* Clean up the coloring (map_colors, scalarmapping_from_data, etc.)
* Move TemporalNetwork to Ensemble NetworkX
* Redo Dataset object and make a .to_anndata method
* Add subset to orindation plot to only some of the points. useful for different markers.
* Autodetect multiheader for read_dataframe and clean it up