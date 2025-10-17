# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "Divi"
copyright = "2025, Qoro Quantum Ltd."
author = "Qoro Quantum Ltd."
release = "0.3.5"

# Add the project root to the Python path so Sphinx can import the modules
sys.path.insert(0, os.path.abspath("../../"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["**.ipynb_checkpoints", "examples/qaoa.ipynb"]

# Prevent execution of notebooks during the build (set to 'always' to run them)
nbsphinx_execute = "never"

# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# Automatically generate stub files for autosummary
autosummary_generate = True

# Set the master document (Sphinx 4.0+ uses root_doc instead of master_doc)
root_doc = "index"

# Include both class and __init__ docstrings
autoclass_content = "both"

# Order members by source order
autodoc_member_order = "bysource"

# Include private members (methods starting with _)
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "no-index": False,
}

# -- Napoleon configuration --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pennylane": ("https://docs.pennylane.ai/en/stable/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}

# Ensure the global toctree is used for navigation
html_sidebars = {"**": ["globaltoc.html", "relations.html", "sourcelink.html"]}

# -- Additional configuration for better module discovery ---------------------

# Add modules to be documented
add_module_names = False
pygments_style = "sphinx"

# Configure autosectionlabel
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Disable module index and search page generation
html_use_modindex = False
html_use_search = False
