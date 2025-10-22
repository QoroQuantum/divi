# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import tomllib

# Load pyproject.toml to extract metadata
with open("../../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
poetry_config = pyproject["tool"]["poetry"]
project_name = poetry_config["name"]


project = project_name.replace("qoro-", "").capitalize()
copyright = "2025, Qoro Quantum Ltd."
author = "Qoro Quantum Ltd."
release = poetry_config["version"]

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
    "sphinx.ext.coverage",  # Documentation coverage
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
    "mitiq": ("https://mitiq.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
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

# GitHub Pages specific configuration
html_baseurl = f"https://{project_name.replace('-', '')}.github.io/divi/"
html_extra_path = []
html_copy_source = False
html_show_sourcelink = False

# Ensure the global toctree is used for navigation
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "relations.html", "sourcelink.html"]
}

# -- Additional configuration for better module discovery ---------------------

# Add modules to be documented
add_module_names = False
pygments_style = "sphinx"

# Configure autosectionlabel
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Disable module index and search page generation
html_use_modindex = False
html_use_search = True

# -- Search configuration ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_search_language = "en"
html_search_options = {
    "dict": "/usr/share/dict/words",
    "type": "default",
}

# -- Coverage extension configuration -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/coverage.html

coverage_show_missing_items = True
coverage_ignore_modules = [
    "divi.extern.*",  # External Cirq/SciPy code
    "divi.__pycache__.*",  # Python cache
    "divi.tests.*",  # Test modules
    "divi.*.tests.*",  # Test modules in subpackages
]
coverage_ignore_functions = [
    "test_*",  # Test functions
    "_*",  # Private functions
    "setup_*",  # Setup functions
    "teardown_*",  # Teardown functions
]
coverage_ignore_classes = [
    "Test.*",  # Test classes
    ".*Test",  # Classes ending in Test
    ".*TestCase",  # Test case classes
]
