Mac OS Installation
===================

Some additional steps beyond the standard installation instructions may be
required for certain Mac OS systems.

The steps detailed in this document have been tested for **Mac OS 10.13.3**.


Contents
========

1. `Requirements`_

   1. `Homebrew`_

2. `Troubleshooting`_

   1. `PyQtGraph`_
   2. `Matplotlib`_

Requirements
============

The following packages are required in order to build PySAP:

1. ``gcc``

  Or any c/c++ compiler that supports OpenMP. (Note that the native Mac
  OS ``clang`` does not)

2. ``cmake``

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

  $ export CC="/usr/local/bin/gcc-7"
  $ export CXX="/usr/local/bin/g++-7"

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
