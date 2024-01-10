"""
Galaxy Image Deconvolution
==========================

.. codeauthor:: Samuel Farrens <samuel.farrens@cea.fr>

In this tutorial we will deconvolve the PSF effects from an example galaxy
image using the PySAP-Astro plug-in.

Import dependencies
-------------------

Import functions from PySAP and
`ModOpt <https://cea-cosmic.github.io/ModOpt/>`_.

"""

import numpy as np
import matplotlib.pyplot as plt
from pysap.data import get_sample_data
from astro.deconvolution.deconvolve import sparse_deconv_condatvu
from modopt.signal.noise import add_noise
from modopt.math.convolve import convolve

# %%
# Load astro data
# ---------------
#
# First, we load some example images from the PySAP sample data sets.

galaxy = get_sample_data('astro-galaxy')
psf = get_sample_data('astro-psf')

# %%
# Then we can show the clean galaxy image that we will attempt to recover.

plt.imshow(galaxy)
plt.show()

# %%
# Generate a noisy observation
# ----------------------------
#
# To simulate an observation we convolve the image with a point spread function
# (PSF) using the ModOpt :py:func:`convolve <modopt.math.convolve.convolve>`
# function. Then we add random Gaussian noise with standard deviation
# ``0.0005`` using the ModOpt
# :py:func:`add_noise <modopt.signal.noise.add_noise>` function.

obs_data = add_noise(convolve(galaxy.data, psf.data), sigma=0.0005)

# %%
# Now we can show the noisy and blurred galaxy image.

plt.imshow(obs_data)
plt.show()

# %%
# Deconvolve
# ----------
#
# We will use the :py:func:`sparse_deconv_condatvu
# <astro.deconvolution.deconvolve.sparse_deconv_condatvu>` function from
# PySAP-Astro to deconvolve the noisy image. We set the maximum number of
# iterations for this function to ``3000``.

deconv_data = sparse_deconv_condatvu(obs_data, psf.data, n_iter=3000)

# %%
# We can show the deconvolved galaxy image.

plt.imshow(deconv_data)
plt.show()

# %%
# Residual
# --------
#
# Next, we calculate the residual of our deconvolved image to get a measure of
# how well the deconvolution process performed.

residual = np.abs(galaxy.data - deconv_data)

# %%
# Finally, we can show the residual.

plt.imshow(residual)
plt.show()

# %%
# .. tip::
#
#  Typically for a denoising problem we are aiming for a residual without any
#  structure, i.e. just noise.
