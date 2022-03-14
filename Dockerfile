FROM continuumio/miniconda3

LABEL Description="PySAP Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

ARG DEBIAN_FRONTEND=noninteractive
ARG CC=gcc-10
ARG CXX=g++-10

RUN apt-get update && \
    apt-get install -y gcc-10 g++-10 && \
    apt-get install -y libgl1 && \
    apt-get install -y libnfft3-dev && \
    apt-get install -y cmake && \
    apt-get clean

RUN git clone https://github.com/CEA-COSMIC/pysap.git

RUN cd pysap && \
    conda env create -f environment.yml

ENV PATH /opt/conda/envs/pysap/bin:$PATH

RUN echo "path: $PATH" && \
    cd pysap && \
    python -m pip install .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
