##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Project descriptions
description = """
Python Interactive Sparse Astronomical Data Analysis Package

* Offers a common API for  astronomical and neuroimaging datasets.
* Offers a Python and fast C++ API.
"""
long_description = """
=======
piSAP
=======

piSAP is a Python package related to sparsity and its application in
astronomical or mediacal data analysis.
This package binds the 'sparse2d' C++ library
that allows sparse decomposition, denoising and deconvolution.
"""

# Main setup parameters
NAME = "piSAP"
ORGANISATION = "CEA"
MAINTAINER = "XXX"
MAINTAINER_EMAIL = "XXX"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "COSMIC webPage"
EXTRAURL = "http://cosmic.cosmostat.org/"
URL = "https://github.com/neurospin/pisap"
DOWNLOAD_URL = "https://github.com/neurospin/pisap"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
Antoine Grigis <antoine.grigis@cea.fr>
Jean-Luc Starck <jl.stark@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
"""
AUTHOR_EMAIL = "XXX"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["pisap"]
REQUIRES = [
    "numpy>=1.11.0",
    "scipy>=0.18.0",
    "pyfits>=3.4",
    "nibabel>=2.1.0",
    "pyqtgraph>=0.10.0",
    #"pysparse>=0.0.1"
]
EXTRA_REQUIRES = {}
