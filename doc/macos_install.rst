Mac OS Installation
===================

Some additional steps beyond the standard installation instructions may be
required for certain macOS systems.

The steps detailed in this document were last tested for **macOS 10.14.1**.


Contents
========

1. `Requirements`_

   1. `Xcode Command Line Tools`_
   2. `Homebrew`_

2. `Troubleshooting`_

   1. `Python3`_
   2. `PyQtGraph`_
   3. `Matplotlib`_
   4. `CFITSIO`_

Requirements
============

The following packages are required in order to build PySAP:

1. ``gcc``

  Or any c/c++ compiler that supports OpenMP. (Note that the native Mac
  OS ``clang`` does not)

2. ``cmake``

Xcode Command Line Tools
------------------------

An essential first step for any developer working on macOS is to install the command line tools. This can be done as follows

.. code-block:: bash

  xcode-select --install

Homebrew
--------

The above listed requirements can be readily installed on Mac OS using |link-to-brew|.

.. |link-to-brew| raw:: html

  <a href="https://brew.sh/"
  target="_blank">Homebrew</a>

.. code-block:: bash

  $ brew install gcc cmake

Note that the commands ``gcc`` and ``g++`` default to ``clang``. Before
installing PySAP you should export the environment variables ``CC`` and ``CXX``.

*e.g.*

.. code-block:: bash

  $ export CC="/usr/local/bin/gcc-8"
  $ export CXX="/usr/local/bin/g++-8"

If you encounter problems re-compiling PySAP following an OS update it may be necessary to uninstall Homebrew and repeat these steps.

Troubleshooting
===============

The following subsections propose solutions to some known issues.

Python 3
--------

For some Python 3 installations (such as Anaconda) the ``Python.h`` header file is
located in a directory called ``python3.Xm`` (where X is the minor version number).
This causes issues with Boost, which looks for this header file in ``python3.X``.

*e.g.*

.. code-block:: bash

  In file included from ./boost/python/detail/prefix.hpp:13:0,
                 from ./boost/python/list.hpp:8,
                 from libs/python/src/list.cpp:5:
  ./boost/python/detail/wrap_python.hpp:50:11: fatal error: pyconfig.h: No such file or directory
  # include <pyconfig.h>
           ^~~~~~~~~~~~
  compilation terminated.

This can be easily solved by exporting the following

.. code-block:: bash

  export CPLUS_INCLUDE_PATH=/PATH-TO-PYTHON/include/python3.Xm


PyQtGraph
---------

Some issues may arise with regards to the installation of ``pyqtgraph``. The
easiest solution to this problem is to install the package using |link-to-conda|.

.. |link-to-conda| raw:: html

  <a href="https://conda.io/docs/"
  target="_blank">Anaconda</a>

.. code-block:: bash

  $ conda install pyqtgraph

Alternatively see the |link-to-pyqt| for help.

.. |link-to-pyqt| raw:: html

  <a href="http://www.pyqtgraph.org/"
  target="_blank">PyQtGraph homepage</a>

Matplotlib
----------

If you see the following error or something similar

.. code-block:: bash

  **RuntimeError**: Python is not installed as a framework...

simply create a ``matplotlibrc`` file and specify a backend.

*e.g.*

.. code-block:: bash

  echo "backend: Agg" >> ~/.matplotlib/matplotlibrc

CFITSIO
-------

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
