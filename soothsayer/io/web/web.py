# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, requests, time

# Soothsayer Utils
from soothsayer.utils import add_objects_to_globals
import soothsayer_utils as syu
functions_from_soothsayer_utils = [
    "read_url",
]
add_objects_to_globals(syu, functions_from_soothsayer_utils, globals(), add_version=True, __all__=None)


# ======
# Future
# ======
# (1) Add ability for `path` arguments to be `pathlib.Path`
# (2) Add ability to process ~ in path
# =============
# Web
# =============
# # Reading html from website
# def read_url(url:str, params=None, **kwargs):
#     """
#     Future:
#     Make wrapper for dynamic html and phantom js
#     """
#     dynamic = False
#     if not dynamic:
#         r = requests.get(url, params=params, **kwargs)
#         return r.text
#     else:
#         print("Not finished. Need to make wrapper for dynamic HTML using PhantomJS", file=sys.stderr)
