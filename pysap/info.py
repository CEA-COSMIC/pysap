# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 6

# Expected by setup.py: string of form "X.Y.Z"
__version__ = '{0}.{1}.{2}'.format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
]

# Project descriptions
description = 'Python Sparse data Analysis Package'
SUMMARY = """
.. container:: summary-carousel

    PySAP is a Python module for **sparse data analysis** that offers:

    * a common API for astronomical and neuroimaging datasets.
    * an accces to 'Sparse2D' using a wrapping or a binding strategy.
    * a user graphical interface to play with the provided functions.
"""
long_description = (
    'PySAP\n\n'
    'PySAP is a Python package related to sparsity and its application in'
    'astronomical or mediacal data analysis.\n'
    'This package binds the "sparse2d" C++ library'
    'that allows sparse decomposition, denoising and deconvolution.\n'
)
# Main setup parameters
NAME = 'python-PySAP'
ORGANISATION = 'CEA'
MAINTAINER = 'Antoine Grigis'
MAINTAINER_EMAIL = 'antoine.grigis@cea.fr'
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = 'COSMIC webPage'
EXTRAURL = 'http://cosmic.cosmostat.org/'
URL = 'https://github.com/CEA-COSMIC/pysap'
DOWNLOAD_URL = 'https://github.com/CEA-COSMIC/pysap'
LICENSE = 'CeCILL-B'
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
Antoine Grigis <antoine.grigis@cea.fr>
Samuel Farrens <samuel.farrens@cea.fr>
Jean-Luc Starck <jl.stark@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""
AUTHOR_EMAIL = 'antoine.grigis@cea.fr'
PLATFORMS = 'Linux,OSX'
ISRELEASE = True
VERSION = __version__
PROVIDES = ['pysap']
REQUIRES = [
    'scipy>=1.5.4',
    'numpy>=1.19.5',
    'matplotlib>=3.3.4',
    'astropy>=4.1',
    'nibabel>=3.2.1',
    'pyqtgraph>=0.11.1',
    'progressbar2>=3.53.1',
    'modopt>=1.5.1',
    'scikit-learn>=0.24.1',
    'PyWavelets>=1.1.1',
]
PREINSTALL_REQUIRES = [
    'pybind11==2.6.2',
    'pyqt5==5.15.4',
]
EXTRA_REQUIRES = {
    'gui': {
        'PySide>=1.2.2',
        # 'python-pypipe>=0.0.1'
    }
}
PLUGINS = [
    'pysap-astro==0.0.1',
    'pysap-mri==0.4.0',
    'pysap-etomo==0.0.1',
]
