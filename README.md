
<img src="logo.png" width=200>

_________________________________

#### Current Version:
*v2020.05.20*

#### Citation:
*Espinoza, *Dupont et al. 2020 (In Review)

#### Case studies, tutorials and usage:
Documentation coming soon. .. ... ..... ........

* [Antimicrobial resistance modeling](https://github.com/jolespin/antimicrobial_resistance_modeling/blob/master/Espinoza-Dupont_et_al_2019/Notebooks/markdown_version/Espinoza-Dupont_et_al_2019.md)
* [Denoising Iris dataset + noise attributes with *Clairvoyance*](tutorials/Notebooks/markdown_versions/Denoising_Iris-plus-Noise_with_Clairvoyance/Denoising_Iris-plus-Noise_with_Clairvoyance.md)

#### Installation:
**Please refer to the [installation manual](install/README.md) for installation details.**  
It will make the installation process *much easier* due to all of the dependencies. 

* <s>conda: https://anaconda.org/jolespin/soothsayer</s>

* pip: https://pypi.org/project/soothsayer


#### Development:
*Soothsayer* is in a developmental stage.  If you're not sure if your installation is a developmental version, check by running: `import soothsayer as sy; print(sy.__developmental__)`.  It is *highly recommended* to [update to the current version](https://github.com/jolespin/soothsayer/tree/master/install#update-to-the-current-release-recommended). 

If you are interested in requesting features or wish to report a bug, please post a GitHub issue prefixed with the tag `[Feature Request]` and `[Bug]`, respectively.

#### Known bugs:
* `statsmodels v0.10.0` has a [bug](https://github.com/statsmodels/statsmodels/issues/5899) that fails to unpickle regression models.  If you will be using `soothsayer.regression`, then install the `v0.10.x` patch for `statsmodels` with the following:

```bash
pip install git+https://github.com/statsmodels/statsmodels.git@maintenance/0.10.x
```


#### Contact:
* Josh L. Espinoza: [jespinoz@jcvi.org](jespinoz@jcvi.org).

<img src ="https://allpistuff.com/wp-content/uploads/2018/07/twitter.c0030826.jpg" width=100>
<img src="https://binstar-static-prod.s3.amazonaws.com/latest/img/AnacondaCloud_logo_green.png" width=300>
