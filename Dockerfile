FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda101 && \
     /opt/conda/bin/conda install -y -c pytorch pytorch torchvision && \
     /opt/conda/bin/conda install -y -c anaconda pytest && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install -y -c conda-forge tensorflow && \
    /opt/conda/bin/conda clean -y --all

WORKDIR /workspace
RUN chmod -R a+w .
