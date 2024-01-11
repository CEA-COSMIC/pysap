# pysap-tutorials

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cea-cosmic/pysap-tutorials/master)

This repository holds demos for examples from various pysap plugins.
Click on the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cea-cosmic/pysap-tutorials/master) to launch a binder.

Navigate into the respective directories and run the examples and see the results.

You dont really need any dependencies, things work on the fly directly, all you need is chrome or firefox to work with jupyter notebooks.

If there are any issues faced in any example, please feel free to open an issue and we will address it as soon as possible!


## Google Colaboratory: In development

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEA-COSMIC/pysap-tutorials/)

We can now run the codes on a google colab. However, this is in development and you may need to run some start up script at begining to ensure the codes run fine.

### Running on Google Colab

1) Click on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEA-COSMIC/pysap-tutorials/).

2) Once inside, choose the appropriate example which you want to run.

3) At the begining, click on `+ Code` and please paste below code and run:
```
!apt install libnfft3-dev
!pip install --upgrade git+https://github.com/cea-cosmic/pysap
!pip install --upgrade git+https://github.com/cea-cosmic/pysap-mri
!pip install --upgrade git+https://github.com/cea-cosmic/modopt
```

This will install all the latest libraries and you can now run all your codes fine. Please ensure you run these codes on every launch as Google Colab in current state does not support persistent kernels or Dockers.

*Note: The codes that use Undecimated Wavelet Transform will not work and we are in process of fixing this.*

Going forward, we plan to enable even GPU versions of codes with which you can run faster on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEA-COSMIC/pysap-tutorials/) as compared to [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cea-cosmic/pysap-tutorials/master)

## Local Environment Setup

While we have setup environments to run things of the shelf on Binder and on Google Colab with some modifications, 
it is still better to setup a local environment for run. Please follow the following instructions to setup your 
own local environment:

   1. Install libnfft3: 
    
        ```sudo apt-get install libnfft3-dev```
   
   2. Install miniconda / anaconda: 
   
        ```
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
        chmod +x miniconda.sh
        ./miniconda.sh -b -p $HOME/miniconda
        export PATH=$HOME/miniconda/bin:$PATH
        ```

   3. Create a Virtual Environment:

        ```
        conda env create -f environment_local.yml
        conda activate pysap-tutorials
        ```
   
   4. Install PySAP and add support to Undecimated Wavelet: Currently we need this step to add support for 
   Undecimated Wavelets, you can skip this step if you dont want to use Undecimated Wavelets.
        ```
        git clone https://github.com/cea-cosmic/pysap
        cd pysap
        python setup build
        export PATH=$PWD/build/temp.linux-x86_64-3.7/extern/bin:$PATH
        ```

   5. Launch jupyter notebook

        ```
        pip install jupyter
        jupyter notebook
        ```
