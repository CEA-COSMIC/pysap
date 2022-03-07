Mac OS Installation
===================

Some additional steps beyond the standard installation instructions may be
required for certain macOS systems.

The steps detailed in this document were last tested for **macOS 10.14.2**.


Contents
--------

1. `Requirements`_

   1. `Xcode Command Line Tools`_
   2. `Homebrew`_

2. `Troubleshooting`_

   1. `PyQtGraph`_
   2. `Matplotlib`_
   3. `CFITSIO`_

Requirements
------------

The following packages are required in order to build PySAP:

1. ``cmake``

2. ``libomp`` (or ``gcc``\*)

  \*Any c/c++ compiler that supports OpenMP should work. (Note that the native macOS
  ``clang`` does not provide OpenMP support on its own)


Xcode Command Line Tools
^^^^^^^^^^^^^^^^^^^^^^^^

An essential first step for any developer working on macOS is to install the command line tools. This can be done as follows

.. code-block:: bash

  xcode-select --install

Homebrew
^^^^^^^^

The above listed requirements can be readily installed on macOS using |link-to-brew|.

.. |link-to-brew| raw:: html

  <a href="https://brew.sh/"
  target="_blank">Homebrew</a>

.. code-block:: bash

  brew install cmake libomp

Note that if you install ``gcc`` the commands ``gcc`` and ``g++`` default to ``clang``.
Therefore, before installing PySAP, you should export the environment variables ``CC`` and ``CXX``.

*e.g.*

.. code-block:: bash

  export CC="/usr/local/bin/gcc-8"
  export CXX="/usr/local/bin/g++-8"

If you encounter problems re-compiling PySAP following an OS update it may be necessary to uninstall Homebrew and repeat these steps.

Troubleshooting
---------------

The following subsections propose solutions to some known issues.

PyQtGraph
^^^^^^^^^

Some issues may arise at runtime with regards to the installation of ``pyqtgraph``. The
easiest solution to this problem is to install the package using |link-to-conda|.

.. |link-to-conda| raw:: html

  <a href="https://conda.io/docs/"
  target="_blank">Anaconda</a>

.. code-block:: bash

  conda install pyqtgraph

Alternatively reinstall PyQT5.

.. code-block:: bash

  pip install -I pyqt5

See the |link-to-pyqt| for further help.

.. |link-to-pyqt| raw:: html

  <a href="http://www.pyqtgraph.org/"
  target="_blank">PyQtGraph homepage</a>

Matplotlib
^^^^^^^^^^

If you see the following error or something similar

.. code-block:: bash

  **RuntimeError**: Python is not installed as a framework...

simply create a ``matplotlibrc`` file and specify a backend.

*e.g.*

.. code-block:: bash

  echo "backend: Agg" >> ~/.matplotlib/matplotlibrc

CFITSIO
^^^^^^^

If you encounter this error

.. code-block:: bash

  configure: error: cannot run C compiled programs.
  If you meant to cross compile, use `--host'.
  See `config.log' for more details
  make[2]: *** [cfitsio/src/cfitsio-stamp/cfitsio-configure] Error 1
  make[1]: *** [CMakeFiles/cfitsio.dir/all] Error 2
  make[1]: *** Waiting for unfinished jobs....

It may be necessary to install the macOS SDK headers. This can be done as follows

.. code-block:: bash

  cd /Library/Developer/CommandLineTools/Packages/
  open macOS_SDK_headers_for_macOS_10.14.pkg
