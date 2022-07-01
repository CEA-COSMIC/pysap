Installation
============

.. attention::

  The installation of PySAP has been extensively tested on Ubuntu and macOS,
  however we cannot guarantee it will work on every operating system. A Docker
  image is available (`see below <#docker-image>`_) for those unable to install
  PySAP directly.

  If you encounter any installation issues be sure to go through the following
  steps before opening a new issue:

  1. Check that that all the `dependencies <dependencies.html>`_ have been
     correctly installed.
  2. Read through all of the documentation provided, including the
     troubleshooting suggestions.
  3. Check if your problem has already been addressed in a previous |link-to-issues|.

Basic installation
------------------

You can install the latest release of PySAP from |link-to-pypi| as follows.

.. tip::
  :class: margin

  Depending on your Python setup you may need to provide the ``--user`` option.

  .. code-block:: bash

    pip install --user python-pysap

.. code-block:: bash

  pip install python-pysap

Developers
----------

Developers are recommend to clone the repository and build the package locally.
To build PySAP locally run the following.

.. code-block:: bash

  pip install .

Alternatively, you can also run the following.

.. code-block:: bash

  python setup.py install

Custom installation
-------------------

.. caution::
  :class: margin

  These options can also be invoked when installing with pip using the
  ``--install-option="<OPTION>"`` option. For example,

  .. code-block:: bash

    pip install . --install-option="--noplugins"

  However, this will disable the use of wheels and may take significantly
  longer to build all of the dependencies. Therefore, when installing PySAP
  this way, it is recommended to pre-install all the required dependencies or
  use the Conda environment provided.

The following options can be passed when running ``python setup.py install``:

- ``--noplugins`` : Install PySAP without any plug-ins
- ``--only=<PLUG-IN NAME>`` : Install PySAP with only the specified plug-in
  name(s) (comma separated)
- ``--nosparse2d`` : Install PySAP without building Sparse2D

.. admonition:: Example

  To install PySAP with only the ETomo plug-in and without Sparse2D
  you would run the following.

  .. code-block:: bash

    python setup.py install --nosparse2d --only=pysap-etomo


.. figure:: https://continentcot.ca/blogue/wp-content/uploads/2017/05/logo_conda_RGB.png
  :figclass: margin
  :width: 200px
  :alt: Conda logo
  :target: https://docs.conda.io/

Conda environment
-----------------

A a |link-to-conda| ``environment.yml`` file is provided to facilitate the
installation of the required PySAP dependencies along with some optional
dependencies that provide additional features. To build the environment run the
following.

.. code-block:: bash

  conda env create -f environment.yml

Then to activate the environment run the following.

.. code-block:: bash

  conda activate pysap

Finally, install PySAP following the instructions above.

.. figure:: https://www.logo.wine/a/logo/Docker_(software)/Docker_(software)-Logo.wine.svg
  :figclass: margin
  :width: 210px
  :alt: Docker logo
  :target: https://www.docker.com/

Docker image
------------

A PySAP |link-to-docker| image is available via |link-to-dockerhub| that
includes the latest stable version of PySAP pre-installed. To install the image
run the following.

.. code-block:: bash

  docker pull ceacosmic/pysap

A Jupyter notebook can be launched using the Docker image as a backend and with
access to the user's current working directory as follows.

.. code-block:: bash

  docker run -p 8888:8888 -v ${PWD}:/home ceacosmic/pysap

Troubleshooting
---------------
If you encounter any difficulties installing PySAP we recommend that you
open a |link-to-issue| and we will do our best to help you.

.. figure:: https://www.logo.wine/a/logo/Linux/Linux-Logo.wine.svg
  :figclass: margin
  :width: 120px
  :alt: Linux logo
  :target: https://www.linux.org/

Linux
^^^^^

PySAP is actively supported and developed using the latest versions of Ubuntu
and CentOS. We are confident that you will be able to install PySAP on these
distributions if all the `required dependencies <dependencies.html>`_ have been
installed.

.. figure:: https://www.logo.wine/a/logo/Apple_Inc./Apple_Inc.-Logo.wine.svg
  :figclass: margin
  :height: 110px
  :alt: Apple Inc. logo
  :target: https://www.apple.com/macos

macOS
^^^^^

PySAP is actively supported and developed using the latest versions of macOS.
We are confident that you will be able to install PySAP on most versions of
this operating system if all the `required dependencies <dependencies.html>`_
have been installed.

.. important::
  :class: margin

  macOS developer tools must be installed beforehand.

  .. code:: bash

    xcode-select --install

The easiest way to install CMake and add OpenMP support for Clang on macOS is
to install the following packages using |link-to-homebrew|.

.. code:: bash

  brew install cmake libomp

Further help with macOS can be found |link-to-macos|.

.. figure:: https://www.logo.wine/a/logo/Microsoft_Windows/Microsoft_Windows-Logo.wine.svg
  :figclass: margin
  :height: 150px
  :alt: Microsoft Windows logo
  :target: https://www.microsoft.com/en-us/windows

Windows
^^^^^^^

.. tip::
  :class: margin

  Windows users are encouraged to use the PySAP Docker image to avoid
  installation issues.

PySAP is not actively supported or developed on Windows, however some tips are
provided in |link-to-windows|.

.. |link-to-issues| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap/issues" target="_blank">issue</a>

.. |link-to-pypi| raw:: html

  <a href="https://pypi.org/project/python-pysap" target="_blank">PyPI</a>

.. |link-to-issue| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap/issues/new/choose"
  target="_blank">new issue</a>

.. |link-to-conda| raw:: html

  <a href="https://docs.conda.io/" target="_blank">Conda</a>

.. |link-to-docker| raw:: html

  <a href="https://www.docker.com/" target="_blank">Docker</a>

.. |link-to-dockerhub| raw:: html

  <a href="https://hub.docker.com/repository/docker/ceacosmic/pysap"
  target="_blank">Docker Hub</a>

.. |link-to-homebrew| raw:: html

  <a href="https://brew.sh/" target="_blank">Homebrew</a>

.. |link-to-macos| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap/blob/master/doc/macos_install.rst
  "target="_blank">here</a>

.. |link-to-windows| raw:: html

  <a href="https://gist.github.com/chaithyagr/4104df91fbebf44fce1589e96baa6eda
  "target="_blank">this Gist</a>
