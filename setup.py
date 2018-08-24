import re
from setuptools import setup

# Version
version = None
with open("./soothsayer/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip()

setup(name='soothsayer',
      version=version,
      description='High-level API for (bio-)informatics',
      url='https://github.com/jolespin/soothsayer',
      author='Josh L. Espinoza',
      author_email='jespinoz@jcvi.org',
      license='MIT',
      packages=["soothsayer"],
      install_requires=[
    # "python >= 3.6",
	"matplotlib >= 2.2.2",
	"scipy >= 1.0",
	"scikit-learn >= 0.19.1",
    'numpy >= 1.9.2, < 1.14.0',
    'pandas >= 0.19.2, < 0.23.0',
	'networkx >= 2.0',
	'ete3 >= 3.0',
	'scikit-bio >= 0.5',
    "biopython >= 1.5",
    "xarray >= 0.10.3",
    "tqdm >=4.19",
      ],
      zip_safe=False)
