# -*- coding: utf-8 -*-
# Python Template sphinx config

# Import relevant modules
import sys
import os
from importlib_metadata import metadata

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ------------------------------------------------

# General information about the project.
project = 'pysap'

mdata = metadata(f'python-{project}')
author = mdata['Author']
version = mdata['Version']
copyright = '20202, {}'.format(author)
gh_user = 'CEA-COSMIC'

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.3'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
    'numpydoc',
]

# Include module names for objects
add_module_names = False

# Set class documentation standard.
autoclass_content = 'class'

# Audodoc options
autodoc_default_options = {
    'member-order': 'bysource',
    'private-members': True,
    'show-inheritance': True
}

# Generate summaries
autosummary_generate = True

# Suppress class members in toctree.
numpydoc_show_class_members = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Sphinx Gallery
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'download_all_examples': False,
    'show_signature': False,
    'image_srcset': ["2x"],
    'reference_url': {
        'pysap': None,
        'astro': None,
        'etomo': None,
        'mri': None
    },
    'thumbnail_size': (500, 500),
    'compress_images': ('images', 'thumbnails'),
}

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'sphinxawesome_theme'
html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'repository_url': 'https://github.com/CEA-COSMIC/pysap',
    'repository_branch': 'develop',
    'use_issues_button': True,
    'use_download_button': False,
    'use_repository_button': True,
    'use_edit_page_button': True,
    'path_to_docs': 'docs/source',
    'home_page_in_toc': True,
    'logo_only': True,
    'home_page_in_toc': False,
}
html_collapsible_definitions = True
html_awesome_headerlinks = True
# html_permalinks_icon = (
#     '<svg xmlns="http://www.w3.org/2000/svg" '
#     'viewBox="0 0 24 24">'
#     '<path d="M3.9 12c0-1.71 1.39-3.1 '
#     "3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 "
#     "5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 "
#     "13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 "
#     "3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 "
#     '5-5s-2.24-5-5-5z"/></svg>'
# )
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f'{project} v{version}'

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '../images/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%d %b, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Intersphinx Mapping ----------------------------------------------

# Refer to the package libraries for type definitions
intersphinx_mapping = {
    'python': ('http://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'progressbar': ('https://progressbar-2.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org', None),
    'astropy': ('http://docs.astropy.org/en/latest/', None),
    'cupy': ('https://docs-cupy.chainer.org/en/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': (
        'http://scikit-learn.org/stable',
        (None, './_intersphinx/sklearn-objects.inv')
    ),
    'tensorflow': (
        'https://www.tensorflow.org/api_docs/python',
        (
            'https://github.com/GPflow/tensorflow-intersphinx/'
            + 'raw/master/tf2_py_objects.inv')
    ),
    'modopt': ('https://cea-cosmic.github.io/ModOpt/', None),

}

# -- BibTeX Setting  ----------------------------------------------

bibtex_bibfiles = ['refs.bib', 'my_ref.bib']
bibtex_default_style = 'alpha'
