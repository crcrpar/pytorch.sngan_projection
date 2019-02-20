FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# create user
ARG user_id
ARG user_name
ARG group_id
ARG group_name
RUN groupadd -g ${group_id} ${group_name} && \
    useradd -u ${user_id} -g ${group_name} -s /bin/bash -m ${user_name} && \
    echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R ${user_name}:${group_name} /home/${user_name}

# Default
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         vim \
         ca-certificates \
         libjpeg-dev \
         libgl1-mesa-dev \
         libpng-dev \
         build-essential \
         zip \
         unzip \
         libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

# Python Anaconda default.
RUN wget -q https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
# Install PyTorch V1.
ENV PATH /opt/conda/bin:$PATH
ARG PYTHON_VERSION
RUN conda install -y python=$PYTHON_VERSION && \
    conda install -y -c conda-forge feather-format && \
    conda install -y -c conda-forge jupyterlab && \
    conda install -y -c conda-forge jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --system
RUN conda install -y pytorch torchvision -c pytorch
RUN conda install -y -c conda-forge tensorflow && \
    conda clean -y --all && \
    pip install --no-cache-dir tensorboardX

ENV PATH /opt/conda/bin:$PATH
WORKDIR /src
