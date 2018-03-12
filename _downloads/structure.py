"""
Anstronomic/Neuroimaging common structure overview
==================================================

Credit: A Grigis

pysap is a Python package related to sparsity and its application in
astronomical or mediacal data analysis. This package also bind the 'sparse2d'
C++ library that allows fast and extended sparse decomposition, denoising and
deconvolution. It is accessible to everybody, and is reusable in various
contexts. The project is hosted on github: https://github.com/neurospin/pysap.

First checks
------------

In order to test if the 'pysap' package is installed on your machine, you can
check the package version
"""

import pysap

print pysap.__version__

#############################################################################
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly:

import pysap.configure

print pysap.configure.info()

#############################################################################
# Import astronomical data
# ------------------------
#
# The package provides a common interface to import and visualize astronomical
# FITS dataset. It also embeds a set of toy dataset that will be used during
# this tutorial:

import pysap
from pprint import pprint
from pysap.data import get_sample_data

image = get_sample_data("astro-fits")
print image.shape, image.spacing, image.data_type
pprint(image.metadata)
print image.data.dtype
image.show()

#############################################################################
# Import neuroimaging data
# ------------------------
#
# The package provides a common interface to import and visualize neuroimaging
# NIFTI dataset. It also embeds a set of toy dataset that will be used during
# this tutorial:

import pysap
from pprint import pprint
from pysap.data import get_sample_data

image = get_sample_data("mri-nifti")
image.scroll_axis = 2
print image.shape, image.spacing, image.data_type
pprint(image.metadata)
print image.data.dtype
image.show()

#############################################################################
# Decompose/recompose an image using a fast ISAP/C++ based transform
# ------------------------------------------------------------------
#
# The package provides also a common interface to the ISAP C++ software
# developped by the COSMOSTAT lab. The code is optimzed and give access to
# many decompsition strategies. All the ISAP library decompositions have
# been declared in a registery:

from pprint import pprint
import pysap

pprint(pysap.AVAILABLE_TRANSFORMS)

#############################################################################
# We illustrate the the decompose/recompose using a 'FastCurveletTransform'
# and 4 scales:

import pysap
from pysap.data import get_sample_data

image = get_sample_data("mri-slice-nifti")
transform_klass = pysap.load_transform("FastCurveletTransform")
transform = transform_klass(nb_scale=4, verbose=1)
transform.data = image
transform.analysis()
transform.show()
rec_image = transform.synthesis()
rec_image.show()
