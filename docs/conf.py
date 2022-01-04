# -*- coding: utf-8 -*-
#
# Automatically generated by nengo-bones, do not edit this file directly

import pathlib

import pytorch_spiking

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "nbsphinx",
    "nengo_sphinx_theme",
    "nengo_sphinx_theme.ext.backoff",
    "nengo_sphinx_theme.ext.sourcelinks",
    "notfound.extension",
    "numpydoc",
    "nengo_sphinx_theme.ext.autoautosummary",
]

# -- sphinx.ext.autodoc
autoclass_content = "both"  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.doctest
doctest_global_setup = """
import pytorch_spiking
import numpy as np
import torch
"""

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "nengo": ("https://www.nengo.ai/nengo/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- nbsphinx
nbsphinx_timeout = -1

# -- notfound.extension
notfound_template = "404.html"
notfound_urls_prefix = "/pytorch-spiking/"

# -- numpydoc config
numpydoc_show_class_members = False

# -- nengo_sphinx_theme.ext.autoautosummary
autoautosummary_change_modules = {
    "pytorch_spiking": [
        "pytorch_spiking.modules.SpikingActivation",
        "pytorch_spiking.modules.Lowpass",
        "pytorch_spiking.modules.TemporalAvgPool",
    ],
}

# -- nengo_sphinx_theme.ext.sourcelinks
sourcelinks_module = "pytorch_spiking"
sourcelinks_url = "https://github.com/nengo/pytorch-spiking"

# -- sphinx
nitpicky = True
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]
linkcheck_timeout = 30
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
linkcheck_ignore = [r"http://localhost:\d+"]
linkcheck_anchors = True
default_role = "py:obj"
pygments_style = "sphinx"
user_agent = "pytorch_spiking"

project = "PyTorchSpiking"
authors = "Applied Brain Research"
copyright = "2020-2022 Applied Brain Research"
version = ".".join(pytorch_spiking.__version__.split(".")[:2])  # Short X.Y version
release = pytorch_spiking.__version__  # Full version, with tags

# -- HTML output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "nengo_sphinx_theme"
html_title = f"PyTorchSpiking {release} docs"
htmlhelp_basename = "PyTorchSpiking"
html_last_updated_fmt = ""  # Default output format (suppressed)
html_show_sphinx = False
html_favicon = str(pathlib.Path("_static", "favicon.ico"))
html_theme_options = {
    "nengo_logo": "",
    "nengo_logo_color": "#a8acaf",
    "tagmanager_id": "GTM-KWCR2HN",
}
