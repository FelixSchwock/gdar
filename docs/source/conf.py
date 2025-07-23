# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup -----------------------------------------------------
# Add project root to sys.path so autodoc can find gdar
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information --------------------------------------------
project = 'GDAR'
author = 'Felix Schwock'
release = '0.1.0'  # Keep this in sync with your pyproject.toml or __version__

# -- General configuration ------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Include documentation from docstrings
    'sphinx.ext.napoleon',      # Support for NumPy/Google-style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx_autodoc_typehints', # Show type hints in docs
]

# Templates path
templates_path = ['_templates']

# Patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output ----------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc settings -----------------------------------------------
autodoc_typehints = 'description'  # Show type hints in the parameter description
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autoclass_content = 'class'  # Show both class docstring and __init__ docstring
