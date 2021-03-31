
<img src="logo/soothsayer_square.png" width=200>

_________________________________
#### Description:
*Soothsayer* is a high-level package for (bio-)informatics with various methods for network analysis, hierarchical ensembles of classifiers, feature selection, plotting, and more.

#### Current Version:
*v2021.03.22*

#### Citation:
Espinoza JL, Dupont CL, O’Rourke A, Beyhan S, Morales P, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. PLOS Computational Biology 17(3): e1008857. [https://doi.org/10.1371/journal.pcbi.1008857](https://doi.org/10.1371/journal.pcbi.1008857)

#### Contact:
Josh L. Espinoza: [jespinoz@jcvi.org](jespinoz@jcvi.org).

_________________________________

#### *Soothsayer* Ecosystem:
* [soothsayer_utils](https://github.com/jolespin/soothsayer_utils) - Utility functions for *Soothsayer*
* [ensemble_networkx](https://github.com/jolespin/ensemble_networkx) - High-level Ensemble Network implementations in Python. Built on top of NetworkX and Pandas.
* [hive_networkx](https://github.com/jolespin/hive_networkx) - High-level Hive plot (Martin Krzywinski et al. 2012) implementations using *Matplotlib* in Python. Built on top of *NetworkX* and *Pandas*.
* [compositional](https://github.com/jolespin/compositional) - Compositional data analysis in Python.

_________________________________

#### Case studies, tutorials and usage:
Documentation will be released upon version 1.0 once API is stabilized.

* [Antimicrobial resistance modeling associated with Espinoza & Dupont et al. 2021](https://github.com/jolespin/antimicrobial_resistance_modeling/blob/master/Espinoza-Dupont_et_al_2021/Notebooks/markdown_version/Espinoza-Dupont_et_al_2021__Supplemental.md)
* [Feature selection using *Clairvoyance* on the Iris dataset with 1000 noise attributes](tutorials/Notebooks/markdown_versions/Denoising_Iris-plus-Noise_with_Clairvoyance/Denoising_Iris-plus-Noise_with_Clairvoyance.md)

_________________________________

#### Installation:
**Please refer to the [installation manual](install/README.md) for installation details.**  
It will make the installation process *much easier* due to all of the Python and R dependencies. 

* pip: [https://pypi.org/project/soothsayer](https://pypi.org/project/soothsayer)

_________________________________


#### Development:
*Soothsayer* is in a developmental stage.  If you're not sure if your installation is a developmental version, check by running: `import soothsayer as sy; print(sy.__developmental__)`.  It is *highly recommended* to [update to the current version](https://github.com/jolespin/soothsayer/tree/master/install#update-to-the-current-release-recommended). 

If you are interested in requesting features or wish to report a bug, please post a GitHub issue prefixed with the tag `[Feature Request]` and `[Bug]`, respectively.

_________________________________


#### `Soothsayer` in the wild:


_________________________________
* Espinoza JL, Dupont CL, O’Rourke A, Beyhan S, Morales P, et al. (2021) Predicting antimicrobial mechanism-of-action from transcriptomes: A generalizable explainable artificial intelligence approach. PLOS Computational Biology 17(3): e1008857. https://doi.org/10.1371/journal.pcbi.1008857


* Yu Imai, Kirsten J. Meyer, Akira Iinishi, Quentin Favre-Godal, Robert Green, Sylvie Manuse, Mariaelena Caboni, Miho Mori, Samantha Niles, Meghan Ghiglieri, Chandrashekhar Honrao, Xiaoyu Ma, Jason Guo, Alexandros Makriyannis, Luis Linares-Otoya, Nils Böhringer, Zerlina G. Wuisan, Hundeep Kaur, Runrun Wu, Andre Mateus, Athanasios Typas, Mikhail M. Savitski, Josh L. Espinoza, Aubrie O’Rourke, Karen E. Nelson, Sebastian Hiller, Nicholas Noinaj, Till F. Schäberle, Anthony D’Onofrio & Kim Lewis. (2019). A new antibiotic selectively kills Gram-negative pathogens. Nature. doi:10.1038/s41586-019-1791-1. 

* Espinoza JL, Shah N, Singh S, Nelson KE, Dupont CL. Applications of weighted association networks applied to compositional data in biology. Environ Microbiol. 2020 May 20;. doi: 10.1111/1462-2920.15091. PubMed PMID: 32436334.


<img src ="https://allpistuff.com/wp-content/uploads/2018/07/twitter.c0030826.jpg" width=100> <img src="logo/soothsayer_wide.png" width=200>


