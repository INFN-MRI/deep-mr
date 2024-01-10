# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"deepmr"
copyright = u"2023, Matteo Cencini"
author = u"Matteo Cencini"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
#    "autoapi.extension",
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["deepinv",
						"h5py",
						"ismrmrd",
						"mat73",
						"matplotlib",
						"nibabel",
						"numpy",
						"numba",
						"pydicom",
						"scipy",
						"torch",
						"tqdm"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/INFN-PREDATOR/deep-mr",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "logo": {
        "image_light": "https://github.com/INFN-PREDATOR/deep-mr/blob/main/docs/source/figures/deepmr_logo.png?raw=true",
        "image_dark": "https://github.com/INFN-PREDATOR/deep-mr/blob/main/docs/source/figures/deepmr_logo_dark.png?raw=true",
        "scale": "10%",
    },

}
