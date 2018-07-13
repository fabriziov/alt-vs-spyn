FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    software-properties-common \
    build-essential

RUN add-apt-repository multiverse

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    nano \
    vim  \
    curl \
    git \
    man \
    unzip \
    wget \
    zlib1g \
    zlib1g-dev \
    llvm-3.7-dev \  
    llvm-3.7 \
    python3-pip \
    libedit-dev \
    python3.5-dev \
    libpython3.5-dev

ENV PYTHONIOENCODING UTF-8
ENV LLVM_CONFIG /usr/bin/llvm-config-3.7
ENV PYTHONPATH = ${PYTHONPATH}:/alt-vs-spyn/

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

RUN pip3 install enum34 argparse ipython numpy==1.12.1

RUN wget https://github.com/fabriziov/llvmlite/archive/v0.9.0.zip && \
    unzip v0.9.0.zip && \
    cd llvmlite-0.9.0/ && \
    python3 setup.py build && \
    python3 setup.py install

RUN pip3 install numba==0.24.0 scipy==0.19.1 scikit-learn==0.18.2

RUN git clone https://github.com/fabriziov/alt-vs-spyn.git
RUN cat alt-vs-spyn/data/eurlex.ts.data.part1of3 alt-vs-spyn/data/eurlex.ts.data.part2of3 alt-vs-spyn/data/eurlex.ts.data.part3of3 > alt-vs-spyn/data/eurlex.ts.data
RUN rm alt-vs-spyn/data/eurlex.ts.data.part1of3 alt-vs-spyn/data/eurlex.ts.data.part2of3 alt-vs-spyn/data/eurlex.ts.data.part3of3
CMD /bin/bash
