# -*- coding: utf-8 -*-
"""
Galaxy Image Denoising
======================

In this tutorial we will remove the noise from an example galaxy image.

Import Dependencies
-------------------

Import functions from PySAP and ModOpt.

"""

import numpy as np
import matplotlib.pyplot as plt
from pysap import Image
from pysap.data import get_sample_data
from astro.denoising.denoise import denoise
from modopt.signal.noise import add_noise

#############################################################################
# Load the image of galaxy NGC2997

galaxy = get_sample_data('astro-ngc2997')

#############################################################################
# Show the clean galaxy image

plt.imshow(galaxy)
plt.show()

#############################################################################
# Generate noisy observation
# --------------------------
#
# Add random Gaussian noise with standard deviation 100 using the `add_noise`
# function.

obs_data = add_noise(galaxy.data, sigma=100)

#############################################################################
# Create a PySAP image object

image_obs = Image(data=np.abs(obs_data))

#############################################################################
# Show the noisy galaxy image

plt.imshow(image_obs)
plt.show()

#############################################################################
# Denoise
# -------
#
# Use the `sparse_deconv_condatvu` function to deconvolve the noisy image and
# set the maximum number of iterations to 3000.

denoise_data = denoise(obs_data, n_scales=4)

#############################################################################
# Create a PySAP image object for the result

image_rec = Image(data=np.abs(denoise_data))

#############################################################################
# Show the deconvolved galaxy image

plt.imshow(image_rec)
plt.show()

#############################################################################
# Residual
# --------
#
# Create a PySAP image object for the residual

residual = Image(data=np.abs(galaxy.data - denoise_data))

#############################################################################
# Show the residual

plt.imshow(residual)
plt.show()
