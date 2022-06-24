"""
PySAP Basics
============

.. codeauthor:: Antoine Grigis <antoine.grigis@cea.fr>

This example introduces some of the basic features of the core PySAP package
as well as some tests to ensure that everything has been installed correctly.

First checks
------------

In order to test if the ``pysap`` package is properly installed on your
machine, you can check the package version.

"""

import matplotlib.pyplot as plt
import pysap

print(pysap.__version__)

# %%
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly:

import pysap.configure

print(pysap.configure.info())

# %%
# Import astronomical data
# ------------------------
#
# PySAP provides a common interface for importing and visualising
# astronomical FITS datasets. A sample of toy datasets is provided that will be
# used in this tutorial.

import pysap
from pprint import pprint
from pysap.data import get_sample_data

image = get_sample_data('astro-fits')
print(image.shape, image.spacing, image.data_type)
pprint(image.metadata)
print(image.data.dtype)
plt.imshow(image)
plt.show()

# %%
# Import neuroimaging data
# ------------------------
#
# PySAP also provides a common interface for importing and visualising
# neuroimaging NIFTI datasets. A sample of toy datasets is provided that will
# be used in this tutorial.

import pysap
from pprint import pprint
from pysap.data import get_sample_data

image = get_sample_data('mri-nifti')
image.scroll_axis = 2
print(image.shape, image.spacing, image.data_type)
pprint(image.metadata)
print(image.data.dtype)
# image.show()  ## uncomment to visualise this object

# %%
# Decompose/recompose an image using a fast ISAP/C++ based transform
# ------------------------------------------------------------------
#
# PySAP includes Python bindings for the Sparse2D C++ library of wavelet
# transforms developped at the COSMOSTAT lab. PySAP also uses the
# PyWavelet package. The code is optimsed and provides access to many
# image decompsition strategies.
#
# All the transforms available can be listed as follows.

from pprint import pprint
import pysap

pprint(pysap.wavelist())
pprint(pysap.wavelist(family='isap-3d'))

# %%
# Here we illustrate the the decomposition/recomposition process using a
# Daubechies (``'db3'``) wavelet from pywt with 4 scales:

import pysap
from pysap.data import get_sample_data

image = get_sample_data('mri-slice-nifti')
transform_klass = pysap.load_transform('db3')
transform = transform_klass(nb_scale=4, verbose=1, padding_mode='symmetric')
transform.data = image
transform.analysis()
# transform.show()  ## uncomment to visualise this object
rec_image = transform.synthesis()
# rec_image.show()  ## uncomment to visualise this object

# %%
# Here we illustrate the the decomposition/recomposition process using a
# fast curvelet transform (``'FastCurveletTransform'``) from Sparse2D with 4
# scales:

image = get_sample_data('mri-slice-nifti')
transform_klass = pysap.load_transform('FastCurveletTransform')
transform = transform_klass(nb_scale=4, verbose=1, padding_mode='zero')
transform.data = image
transform.analysis()
# transform.show()  ## uncomment to visualise this object
rec_image = transform.synthesis()
# rec_image.show()  ## uncomment to visualise this object
