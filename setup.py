import re, datetime
from setuptools import setup, find_packages

# Version
version = None
with open("./soothsayer/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')

setup(name='soothsayer',
      version=version,
      description='High-level API for (bio-)informatics',
      url='https://github.com/jolespin/soothsayer',
      author='Josh L. Espinoza',
      author_email='jespinoz@jcvi.org',
      license='BSD',
      packages=find_packages(include=("*", "./*")),
      install_requires=[
    # "python >= 3.6",
	"matplotlib >= 2.2.2",
	"scipy >= 1.0",
	"scikit-learn >= 0.20.2",
    "numpy >= 1.13", #, < 1.14.0",
    # "numpy >= 1.13",
    'pandas >= 0.24.2',
	'networkx >= 2.0',
	'ete3 >= 3.0',
	'scikit-bio >= 0.5.1',
    # "scikikit-bio >= 0.5.1",
    "biopython >= 1.5",
    "xarray >= 0.10.3",
    "tqdm >=4.19",
    "openpyxl >= 2.5",
    "astropy >= 3.0",
    "rpy2 >= 2.9",
    "matplotlib_venn",
    "palettable >= 3.0.0",
      ],
     include_package_data=True,
     scripts=['bin/clairvoyance.py', "bin/run_soothsayer.py"],


)
