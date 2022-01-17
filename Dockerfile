FROM continuumio/miniconda3

LABEL Description="PySAP Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

ARG DEBIAN_FRONTEND=noninteractive
ARG CC=gcc-8
ARG CXX=g++-8
ARG CPLUS_INCLUDE_PATH=/opt/conda/envs/pysap/include/python3.8/

RUN apt update && \
    apt install -y gcc-8 g++-8 && \
    apt install -y libgl1 && \
    apt install -y libnfft3-dev && \
    apt install -y cmake && \
    apt clean

RUN git clone https://github.com/CEA-COSMIC/pysap.git

RUN cd pysap && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/pysap/bin:$PATH

RUN echo "path: $PATH" && \
    cd pysap && \
    python setup.py install

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
