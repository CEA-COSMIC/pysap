"""
Galaxy Image Deconvolution
==========================

Credit: S. Farrens

In this tutorial we will deconvolve the PSF effects from an example galaxy
image.

Import Dependencies
-------------------

Import functions from PySAP and ModOpt.

"""

import numpy as np
from pysap import Image
from pysap.data import get_sample_data
from pysap.plugins.astro.deconvolution.deconvolve import sparse_deconv_condatvu
from modopt.signal.noise import add_noise
from modopt.math.convolve import convolve

#############################################################################
# Load astro data
# ---------------
#
# Load the example images

galaxy = get_sample_data('astro-galaxy')
psf = get_sample_data('astro-psf')

#############################################################################
# Show the clean galaxy image

galaxy.show()

#############################################################################
# Generate noisy observation
# --------------------------
#
# Convolve the image with a point spread function (PSF) using the `convolve`
# function. Then add random Gaussian noise with standard deviation 0.0005
# using the `add_noise` function.

obs_data = add_noise(convolve(galaxy.data, psf.data), sigma=0.0005)

#############################################################################
# Create a PySAP image object

image_obs = Image(data=np.abs(obs_data))

#############################################################################
# Show the noisy galaxy image

image_obs.show()

#############################################################################
# Deconvolve
# ----------
#
# Use the `sparse_deconv_condatvu` function to deconvolve the noisy image and
# set the maximum number of iterations to 3000.

deconv_data = sparse_deconv_condatvu(obs_data, psf.data, n_iter=3000)

#############################################################################
# Create a PySAP image object for the result

image_rec = Image(data=np.abs(deconv_data))

#############################################################################
# Show the deconvolved galaxy image

image_rec.show()

#############################################################################
# Residual
# --------
#
# Create a PySAP image object for the residual

residual = Image(data=np.abs(galaxy.data - deconv_data))

#############################################################################
# Show the residual

residual.show()
