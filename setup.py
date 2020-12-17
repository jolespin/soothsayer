import re, datetime
from setuptools import setup, find_packages

# Version
version = None
with open("./soothsayer/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in soothsayer/__init__.py"

setup(name='soothsayer',
      version=version,
      description='High-level package for (bio-)informatics',
      url='https://github.com/jolespin/soothsayer',
      author='Josh L. Espinoza',
      author_email='jespinoz@jcvi.org',
      license='BSD-3',
      packages=find_packages(include=("*", "./*")),
      install_requires=[
	"matplotlib >= 3",
    "seaborn >= 0.10.1",
	"scipy >= 1.0",
	"scikit-learn >= 0.20.2",
    "numpy >= 1.13", #, < 1.14.0",
    'pandas >= 1.0',
	'networkx >= 2.0',
	'ete3 >= 3.0',
	'scikit-bio >= 0.5.1',
    "biopython >= 1.5",
    "xarray >= 0.10.3",
    "tqdm >=4.19",
    "openpyxl >= 2.5",
    # "rpy2 >= 2.9.4", 
    "rpy2 >= 3.3.2",
    "matplotlib_venn",
    "palettable >= 3.0.0",
    "adjustText",
    "tzlocal",
    "statsmodels >= 0.10.0", #https://github.com/statsmodels/statsmodels/issues/5899 May need to use this separately: pip install git+https://github.com/statsmodels/statsmodels.git@maintenance/0.10.x
    "mmh3",
    # Extensions
    "soothsayer_utils >= 2020.12.15",
    "compositional >= 2020.12.16",
    "hive_networkx >= 2020.8.3",
    "ensemble_networkx >= 2020.8.24",

    # Optional
    # "teneto",
    # "gneiss", # Removed dependency and moved gneiss as optional in compositional
    # Deprecated
    # "astropy >= 3.0", # Removed depenency and reimplemented biweight midcorrelation

      ],
     include_package_data=True,
     scripts=['bin/clairvoyance.py', "bin/run_soothsayer.py"],


)
