I've tried to make installation as seemless as possible.  There are several R dependencies that are difficult to install via conda which is why there is no conda installation...yet [but I'm working on it :)]  

Current installation method:

```bash
# Create a conda environment to run soothsayer [Default: soothsayer_env]
bash install_soothsayer_environment.sh

# Create a conda environment to run soothsayer with custom environment name
bash install_soothsayer_environment.sh hellosoothsayer_env

# Create a conda environment to run soothsayer with custom environment name & a particular Python version (when I created this the 3.7.x dependencies weren't ready and I've only tested this on 3.6.x)
bash install_soothsayer_environment.sh hellosoothsayer_env 3.6.6
```

Quite a bit of troubleshooting was required to get all of the packages to install (again, the R packages like WGCNA & devtools were tricky) so let me know if you have any issues before creating an issue on GitHub (jespinoz@jcvi.org).
