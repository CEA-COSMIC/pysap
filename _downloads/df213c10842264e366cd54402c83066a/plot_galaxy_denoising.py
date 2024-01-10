"""
Galaxy Image Denoising
======================

.. codeauthor:: Samuel Farrens <samuel.farrens@cea.fr>

In this tutorial we will remove the noise from an example galaxy image using
the PySAP-Astro plug-in.

Import dependencies
-------------------

Import functions from PySAP and
`ModOpt <https://cea-cosmic.github.io/ModOpt/>`_.

"""

import numpy as np
import matplotlib.pyplot as plt
from pysap.data import get_sample_data
from astro.denoising.denoise import denoise
from modopt.signal.noise import add_noise

# %%
# Clean image
# -----------
#
# First, we load the image of galaxy NGC2997 from the PySAP sample data sets.

galaxy = get_sample_data('astro-ngc2997')

# %%
# Then we can show the clean galaxy image that we will attempt to recover.

plt.imshow(galaxy)
plt.show()

# %%
# Generate a noisy observation
# ----------------------------
#
# To simulate an observation of NGC2997 we add random Gaussian noise with
# standard deviation ``100`` using the
# ModOpt :py:func:`add_noise <modopt.signal.noise.add_noise>` function.

obs_data = add_noise(galaxy.data, sigma=100)

# %%
# Now we can show the noisy galaxy image.

plt.imshow(obs_data)
plt.show()

# %%
# .. note::
#
#  :math:`\sigma=100` is quite excesive, but we can more easily visualise the
#  noise added to the image in this example.

# %%
# Denoise
# -------
#
# We will use the :py:func:`denoise <astro.denoising.denoise.denoise>` function
# from PySAP-Astro to denoise the noisy image. We set the number of wavelet
# scales to ``4``.

denoised_data = denoise(obs_data, n_scales=4)

# %%
# We can show the denoised galaxy image.

plt.imshow(denoised_data)
plt.show()

# %%
# Residual
# --------
#
# Next, we calculate the residual of our denoised image to get a measure of
# how well the denoising process performed.

residual = np.abs(galaxy.data - denoised_data)

# %%
# Finally, we can show the residual.

plt.imshow(residual)
plt.show()

# %%
# .. tip::
#
#  Typically for a denoising problem we are aiming for a residual without any
#  structure, i.e. just noise.
