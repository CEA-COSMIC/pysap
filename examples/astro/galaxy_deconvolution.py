"""
Galaxy Image Deconvolution
==========================

Credit: S. Farrens

In this tutorial we will deconvolve the PSF effects from an example galaxy
image.

Import astro data
-----------------

The example galaxy image is convolved with the example PSF and random noise is
added simulating an observation with SNR~5.

"""

import numpy as np
from pisap import Image
from pisap.plugins.astro.deconvolve.deconvolve import sparse_deconv_condatvu
from modopt.signal.noise import add_noise
from modopt.math.convolve import convolve

# Load the example images
galaxy = np.load('/Users/sfarrens/Desktop/example_galaxy_image.npy')
psf = np.load('/Users/sfarrens/Desktop/example_psf_image.npy')

# Show the clean galaxy image
image_true = Image(data=np.abs(galaxy))
image_true.show()

# Generate a noise observed image
obs_data = add_noise(convolve(galaxy, psf), sigma=0.0005)
image_obs = Image(data=np.abs(obs_data))
image_obs.show()

# Deconvolve the observed image
deconv_data = sparse_deconv_condatvu(obs_data, psf)
image_rec = Image(data=np.abs(deconv_data))
image_rec.show()

# Show the residual
residual = Image(data=np.abs(galaxy - deconv_data))
residual.show()
