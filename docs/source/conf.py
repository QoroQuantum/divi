# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import enum
import os
import sys
import tomllib
from datetime import datetime

# Load pyproject.toml to extract metadata
with open("../../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)
project_config = pyproject["project"]
project_name = project_config["name"]


project = project_name.replace("qoro-", "").capitalize()
copyright = f"{datetime.now().year}, Qoro Quantum Ltd."
author = "Qoro Quantum Ltd."
release = project_config["version"]

# Add the project root to the Python path so Sphinx can import the modules
sys.path.insert(0, os.path.abspath("../../"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",  # Documentation coverage
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.automodapi",  # Generates API pages with proper canonical resolution
    "sphinx_automodapi.smart_resolver",  # Resolves re-exported symbols to their public path
    "sphinx_copybutton",  # "Copy" button on code blocks
    "sphinxcontrib.mermaid",
]

# sphinxcontrib.spelling requires libenchant-2-dev at the system level.
# Only load it when explicitly requested (``make spelling`` sets SPELLING=1)
# so that ``make build`` works on machines without enchant installed.
if os.environ.get("SPELLING"):
    extensions.append("sphinxcontrib.spelling")

templates_path = []
exclude_patterns = []

# Enable nitpick mode by default so broken cross-references surface locally,
# in CI, and on Read the Docs without needing the ``-n`` command-line flag.
nitpicky = True

# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

root_doc = "index"
autoclass_content = "both"
autodoc_default_options = {"member-order": "bysource", "exclude-members": "__weakref__"}


# -- sphinx-automodapi configuration -----------------------------------------
# https://sphinx-automodapi.readthedocs.io/
#
automodapi_toctreedirnm = "api_reference/generated"
automodapi_inheritance_diagram = False  # divi doesn't pull in graphviz
# ``False`` avoids documenting inherited methods in each subclass page, which
# would otherwise duplicate method registrations (e.g. ``update``/``info``
# from ``ProgressReporter`` appearing in both parent and child stub pages).
automodsumm_inherited_members = False

# -- sphinx-copybutton configuration -----------------------------------------
# https://sphinx-copybutton.readthedocs.io/
#
# Strip REPL prompts, shell prompts, and Jupyter ``In [...]`` markers from
# copied code blocks. Matches the pattern used by PennyLane's docs.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Render unions as ``X | Y`` instead of emitting a ``:py:data:`typing.Union```
# cross-ref. Avoids the domain mismatch where Python's intersphinx registers
# ``typing.Union`` as ``py:class`` but the extension emits ``py:data`` on < 3.14.
always_use_bars_union = True

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
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "optional": ":obj:`optional`",
    "array-like": ":term:`array-like`",
    "array_like": ":term:`array-like`",
    "callable": ":py:func:`callable`",
    "OptimizeResult": ":class:`~scipy.optimize.OptimizeResult`",
}
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pennylane": ("https://docs.pennylane.ai/en/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "qiskit_aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "mitiq": ("https://mitiq.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "rich": ("https://rich.readthedocs.io/en/latest/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),
}

# -- Nitpick: suppress cross-reference warnings ---------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore_regex
#
# Keep this list minimal. Before adding a new entry, first try to resolve the
# underlying issue: ``__all__`` in submodules, intersphinx mapping for the
# library, or a fully-qualified cross-reference. Every entry here is debt.
nitpick_ignore_regex = [
    # cirq uses Google-style docs (no objects.inv).
    (r"py:class", r"cirq\..*"),
    # dimod's inventory lives under the Ocean umbrella.
    (r"py:class", r"dimod\..*"),
    # TypeVars in ``Generic[InT, OutT]`` — not documentable by Sphinx.
    (r"py:obj", r"divi\.pipeline\.abc\.(InT|OutT)"),
    # ``numpy.float64`` is ``py:attribute`` in numpy's inventory, but
    # ``sphinx-autodoc-typehints`` emits ``py:class``. Domain mismatch.
    (r"py:class", r"numpy\.(float|int|uint)\d+"),
    # Short aliases emitted by ``sphinx-autodoc-typehints`` from runtime
    # type annotations (``np`` = ``numpy``, ``npt`` = ``numpy.typing``).
    # These appear in automodapi-generated stub headings at ``:2`` where
    # the source type hint used the alias form.
    (r"py:class", r"np\..*"),
    (r"py:class", r"npt\..*"),
    # ``scipy.optimize.OptimizeResult`` — bare name from autodoc-typehints.
    (r"py:class", r"^OptimizeResult$"),
]

# ``sphinx-autodoc-typehints`` emits its own warning category for unresolved
# forward references (e.g. cirq's ``OP_TREE`` that is only defined under
# ``TYPE_CHECKING``). It is a third-party library limitation, not a nitpick.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


# Favicon configuration
html_favicon = "_static/favicon.ico"

# Mermaid: tighter subgraph and diagram spacing, smaller font
mermaid_init_config = {
    "startOnLoad": False,
    "themeVariables": {
        "fontSize": "12px",
    },
    "flowchart": {
        "nodeSpacing": 25,
        "rankSpacing": 20,
        "diagramPadding": 8,
        "subGraphTitleMargin": {"top": 0, "bottom": 0},
    },
}

# Theme options
html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#CC3366",
        "color-brand-content": "#CC3366",
        "font-stack": "Poppins, sans-serif",
        "font-stack--headings": "Poppins, sans-serif",
        "font-stack--monospace": "ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Consolas, 'Liberation Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FDB9FD",
        "color-brand-content": "#FDB9FD",
        "color-background-primary": "#131313",
        "color-background-secondary": "#1a1a1a",
        "color-foreground-primary": "#E9EEF6",
    },
}

# Canonical URL: use RTD's env var when building there, empty otherwise
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# GitHub Pages extension (creates .nojekyll / CNAME — not needed on RTD)
if not os.environ.get("READTHEDOCS"):
    extensions.append("sphinx.ext.githubpages")
html_extra_path = []
html_copy_source = False
html_show_sourcelink = False

add_module_names = False
pygments_style = "sphinx"
pygments_dark_style = "monokai"

html_search_language = "en"

# -- Coverage extension configuration -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/coverage.html

coverage_show_missing_items = True
coverage_ignore_modules = [
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

# -- Linkcheck configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-linkcheck-builder

linkcheck_ignore = [
    r"http://localhost:\d+/",
]

# Timeout settings to prevent linkcheck from hanging
linkcheck_timeout = 30  # Timeout in seconds for each link check
linkcheck_retries = 2  # Number of retries for failed links
linkcheck_workers = 5  # Number of concurrent worker threads
linkcheck_rate_limit_timeout = 60  # Maximum wait time (seconds) when rate-limited

# -- Spelling configuration --------------------------------------------------
# https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html

spelling_word_list_filename = "spelling_wordlist.txt"
spelling_show_suggestions = True
spelling_suggestions_path = "spelling_suggestions.txt"
spelling_ignore_acronyms = True
spelling_ignore_pypi_package_names = True
spelling_ignore_python_builtins = True
spelling_ignore_importable_modules = True

# Add filters to ignore common patterns in documentation
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.WikiWordFilter",
    "enchant.tokenize.MentionFilter",
    "enchant.tokenize.HashtagFilter",
]


def autodoc_process_signature(
    _app, what, _name, obj, _options, signature, return_annotation
):
    """Hide enum constructor signature."""
    if what == "class":
        try:
            if issubclass(obj, enum.Enum):
                return "", None
        except TypeError:
            pass
    return signature, return_annotation


def setup(app):
    """Register the autodoc hook."""
    app.connect("autodoc-process-signature", autodoc_process_signature)
