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
from pysap import Image
from pysap.data import get_sample_data
from pysap.plugins.astro.deconvolve.deconvolve import sparse_deconv_condatvu
from modopt.signal.noise import add_noise
from modopt.math.convolve import convolve

# Load the example images
galaxy = get_sample_data('astro-galaxy')
psf = get_sample_data('astro-psf')

# Show the clean galaxy image
galaxy.show()

# Generate a noisy observed image
obs_data = add_noise(convolve(galaxy.data, psf.data), sigma=0.0005)
image_obs = Image(data=np.abs(obs_data))
image_obs.show()

# Deconvolve the observed image
deconv_data = sparse_deconv_condatvu(obs_data, psf.data, n_iter=3000)
image_rec = Image(data=np.abs(deconv_data))
image_rec.show()

# Show the residual
residual = Image(data=np.abs(galaxy.data - deconv_data))
residual.show()
