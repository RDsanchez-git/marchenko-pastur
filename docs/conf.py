import os
import sys

# Apunta al src layout
sys.path.insert(0, os.path.abspath("../src"))

project = "marchenko_pastur"
author = "Roberto Daniel Sánchez"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# ---------------------------------------------------
# Rutas y Exclusiones (Higiene de Build)
# ---------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------
# Configuración Numpydoc y Autodoc
# ---------------------------------------------------
autosummary_generate = True
autosummary_imported_members = True

numpydoc_show_class_members = False
numpydoc_xref_param_type = True  # Cross-referencing automático hacia NumPy/SciPy

autodoc_member_order = "bysource"
add_module_names = False

autodoc_default_options = {
    'exclude-members': 'maketrans',
}

# ---------------------------------------------------
# Cross-Referencing Externo
# ---------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------
# Configuración HTML / UI
# ---------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} v{release}"
html_static_path = ["_static"]

# ---------------------------------------------------
# Sphinix Hook
# ---------------------------------------------------

suppress_warnings = ["myst.xref_missing"]

def autodoc_skip_member(app, what, name, obj, skip, options):
    """Intercepta y omite métodos C-native incompatibles con Sphinx."""
    if name == "maketrans":
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)

