Quickstart Tutorial
===================

.. seealso::

  See our gallery of `examples <auto_examples/index.html>`_ for specific
  applications of PySAP.

Basics
------

You can import the PySAP package as follows.

.. code-block:: python

  import pysap

Then you can print some basic information about PySAP,

.. code-block:: python

  print(pysap.info())

which should look something like this.

.. code-block::

                   .|'''|       /.\      '||'''|,
                   ||          // \\      ||   ||
  '||''|, '||  ||` `|'''|,    //...\\     ||...|'
   ||  ||  `|..||   .   ||   //     \\    ||
   ||..|'      ||   |...|' .//       \\. .||
   ||       ,  |'
  .||        ''

  Package version: 0.0.6

  License: CeCILL-B

  Authors:

  Antoine Grigis <antoine.grigis@cea.fr>
  Samuel Farrens <samuel.farrens@cea.fr>
  Jean-Luc Starck <jl.stark@cea.fr>
  Philippe Ciuciu <philippe.ciuciu@cea.fr>

  Dependencies:

  scipy          : >=1.5.4   - required | 1.7.3     installed
  numpy          : >=1.19.5  - required | 1.22.1    installed
  matplotlib     : >=3.3.4   - required | 3.5.1     installed
  astropy        : >=4.1     - required | 5.0       installed
  nibabel        : >=3.2.1   - required | 3.2.1     installed
  pyqtgraph      : >=0.11.1  - required | 0.12.3    installed
  progressbar2   : >=3.53.1  - required | ?         installed
  modopt         : >=1.5.1   - required | 1.6.0     installed
  scikit-learn   : >=0.24.1  - required | ?         installed
  pywt           : >=1.1.1   - required | 1.2.0     installed
  pysparse       : >=0.0.1   - required | 0.1.0     installed

Sample data
-----------

PySAP includes example data from various imaging domains that can be used for
testing. Sample data sets can downloaded using the
:py:func:`get_sample_data <pysap.data.get_sample_data>` function.

.. admonition:: Example

  To download an image of galaxy NGC2997 you would run the following.

  .. code:: python

    from pysap.data import get_sample_data
    galaxy = get_sample_data('astro-ngc2997')

Image objects
-------------

The principal data type in PySAP are :py:class:`Image <pysap.base.image.Image>`
objects. These objects include a built-in
:py:meth:`show <pysap.base.image.Image.show>` method.

.. admonition:: Example

  Running ``type(galaxy)`` with the object from the previous example should
  give you ``<class 'pysap.base.image.Image'>`` and thus you can directly
  visualise this object.

  .. code:: python

    galaxy.show()

``Image`` objects can easily be created from
:py:class:`Numpy arrays <numpy.ndarray>`.

.. admonition:: Example

  To create a PySAP ``Image`` object from a 2D numpy array you would do the
  following.

  .. code:: python

    import numpy as np
    from pysap import Image
    my_array = np.arange(9).reshape((3, 3))
    my_image = Image(data=my_array)

Transforms
----------

You can display all of the available data transforms (i.e. wavelets and more)
using the :py:func:`wavelist <pysap.utils.wavelist>` function.

.. attention::
  :class: margin

  We plan to provide more detailed information on each of the available
  transforms in the near future. The keys ``'isap-3d'`` and ``'isap-2d'``
  from this dictionary correspond to |link-to-sparse2d|, while ``'pywt'``
  corresponds to |link-to-pywavelet|.

.. code-block:: python

  print(pysap.wavelist())

A given PySAP transform can be loaded using the
:py:func:`load_transform <pysap.utils.load_transform>` function.

.. admonition:: Example

  To load the isotropic undecimated wavelet (or starlet) transform with ``4``
  scales and symmetric padding you would do the following.

  .. code:: python

    from pysap import load_transform
    my_transform = load_transform('BsplineWaveletTransformATrousAlgorithm')(
        nb_scale=4,
        padding_mode='symmetric',
    )

This will create a class instance of the desired transform with parent
:py:class:`WaveletTransformBase <pysap.base.transform.WaveletTransformBase>`,
which has the methods
:py:meth:`analysis <pysap.base.transform.WaveletTransformBase.analysis>` and
:py:meth:`synthesis <pysap.base.transform.WaveletTransformBase.synthesis>`
that can be used to transform images into the corresponding domain and back
again.

.. tip::
  :class: margin

  Transform objects also have access to the ``Image`` ``show`` method. Try
  this!

  .. code:: python

    my_transform.show()

.. admonition:: Example

  Using the transform defined in the previous example and and the ``galaxy``
  ``Image`` object from earlier you can transform this data into the wavelet
  domain as follows.

  .. code:: python

    my_transform.data = galaxy
    my_transform.analysis()



.. |link-to-sparse2d| raw:: html

  <a href="https://github.com/CosmoStat/Sparse2D" target="_blank">Sparse2D</a>

.. |link-to-pywavelet| raw:: html

  <a href="https://pywavelets.readthedocs.io/ "target="_blank">PyWavelets</a>
