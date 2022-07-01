Dependencies
============

.. attention::
  :class: margin

  PySAP will automatically install all of the required Python dependencies,
  however issues may arise on some operating systems. If you encounter any
  problems please ensure that you have all of the listed dependencies
  installed before opening a new issue.

.. tip::
  :class: margin

  All of the required dependencies can also be installed using the Conda
  environment provided.

  .. code:: bash

    conda env create -f environment.yml
    conda activate pysap

PySAP requires the following dependencies in order to
`install <installation.html>`_ and run correctly.

1. A C/C++ compiler that supports |link-to-omp|
2. |link-to-cmake| [>=3.0]
3. The latest version of |link-to-modopt|
4. The following third-party Python packages:

   - astropy [>=4.1]
   - matplotlib [>=3.3.4]
   - nibabel [>=3.2.1]
   - numpy [>=1.19.5]
   - scipy [>=1.5.4]
   - scikit-learn [>=0.24.1]
   - progressbar2 [>=3.53.1]
   - pybind11 [==2.6.2]
   - pyqt5 [==5.15.4]
   - pyqtgraph [>=0.11.1]
   - PyWavelets [>=1.1.1]


.. |link-to-omp| raw:: html

 <a href="https://www.openmp.org/" target="_blank">OpenMP</a>

.. |link-to-cmake| raw:: html

 <a href="https://cmake.org/" target="_blank">CMake</a>

.. |link-to-modopt| raw:: html

  <a href="https://cea-cosmic.github.io/ModOpt" target="_blank">ModOpt</a>
