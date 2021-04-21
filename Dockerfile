FROM continuumio/miniconda3

LABEL Description="PySAP Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

ARG DEBIAN_FRONTEND=noninteractive
ARG CC=gcc-8
ARG CXX=g++-8
ARG CPLUS_INCLUDE_PATH=/opt/conda/envs/pysap/include/python3.8/

RUN apt-get update && \
    apt-get install -y gcc-8 g++-8 && \
    apt-get install -y libgl1 && \
    apt-get install -y libnfft3-dev && \
    apt-get install -y cmake && \
    apt-get clean

RUN git clone https://github.com/CEA-COSMIC/pysap.git && \
    cd pysap && \
    git fetch origin pull/150/head:pr-150 && \
    git checkout pr-150 && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/pysap/bin:$PATH

RUN echo "path: $PATH" && \
    cd pysap && \
    python setup.py install

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
