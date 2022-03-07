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
copyright = '2020, {}'.format(author)
gh_user = 'sfarrens'

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
    'sphinxawesome_theme',
    'sphinxcontrib.bibtex',
    'myst_parser',
    'nbsphinx',
    'nbsphinx_link',
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

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinxawesome_theme'
# html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "nav_include_hidden": True,
    "show_nav": True,
    "show_breadcrumbs": True,
    "breadcrumbs_separator": "/",
    "show_prev_next": True,
    "show_scrolltop": True,

}
html_collapsible_definitions = True
html_awesome_headerlinks = True
html_logo = '../images/logo.png'
html_permalinks_icon = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'viewBox="0 0 24 24">'
    '<path d="M3.9 12c0-1.71 1.39-3.1 '
    "3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 "
    "5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 "
    "13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 "
    "3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 "
    '5-5s-2.24-5-5-5z"/></svg>'
)
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, version)

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%d %b, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for nbshpinx output ------------------------------------------


# Custom fucntion to find notebooks, create .nblink files and update the
# notebooks.rst file
def add_notebooks(nb_path='../../notebooks'):

    print('Looking for notebooks')
    nb_ext = '.ipynb'
    nb_rst_file_name = 'notebooks.rst'
    nb_link_format = '{{\n   "path": "{0}/{1}"\n}}'

    nbs = sorted([nb for nb in os.listdir(nb_path) if nb.endswith(nb_ext)])

    for list_pos, nb in enumerate(nbs):

        nb_name = nb.rstrip(nb_ext)

        nb_link_file_name = nb_name + '.nblink'
        print('Writing {0}'.format(nb_link_file_name))
        with open(nb_link_file_name, 'w') as nb_link_file:
            nb_link_file.write(nb_link_format.format(nb_path, nb))

        print('Looking for {0} in {1}'.format(nb_name, nb_rst_file_name))
        with open(nb_rst_file_name, 'r') as nb_rst_file:
            check_name = nb_name not in nb_rst_file.read()

        if check_name:
            print('Adding {0} to {1}'.format(nb_name, nb_rst_file_name))
            with open(nb_rst_file_name, 'a') as nb_rst_file:
                if list_pos == 0:
                    nb_rst_file.write('\n')
                nb_rst_file.write('   {0}\n'.format(nb_name))

    return nbs


# Add notebooks
add_notebooks()

binder = 'https://mybinder.org/v2/gh'
binder_badge = 'https://mybinder.org/badge_logo.svg'
github = 'https://github.com/'
github_badge = 'https://badgen.net/badge/icon/github?icon=github&label'

# Remove promts and add binder badge
nb_header_pt1 = r'''
{% if env.metadata[env.docname]['nbsphinx-link-target'] %}
{% set docpath = env.metadata[env.docname]['nbsphinx-link-target'] %}
{% else %}
{% set docpath = env.doc2path(env.docname, base='docs/source/') %}
{% endif %}

.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>

'''
nb_header_pt2 = (
    r'''    <p><div class="inline-block">'''
    r'''<a href="{0}/{1}/{2}/'''.format(binder, gh_user, project) +
    r'''master?filepath={{ docpath }}">''' +
    r'''<img alt="Binder badge" src="{0}" '''.format(binder_badge) +
    r'''style="vertical-align:text-bottom"></a></div>'''
    r'''<div class="inline-block"><a href=''' +
    r'''"{0}/{1}/{2}/blob/master/'''.format(github, gh_user, project) +
    r'''{{ docpath }}"><img alt="GitHub badge" ''' +
    r'''src="{0}" style="vertical-align:text-bottom">'''.format(github_badge) +
    r'''</a></div></p>'''
)

nbsphinx_prolog = nb_header_pt1 + nb_header_pt2

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
    )

}

# -- BibTeX Setting  ----------------------------------------------

bibtex_bibfiles = ['refs.bib', 'my_ref.bib']
bibtex_default_style = 'alpha'
