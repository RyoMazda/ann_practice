FROM python:3.9.5-slim-buster as builder
# Who uses conda!?

RUN apt-get update && apt-get install -y \
  git wget \
  make libssl-dev gcc g++ \
  libblas-dev liblapack-dev swig \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# Install CMake
ARG CMAKE_VERSION=3.20.3
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar -zxvf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
        ./bootstrap && make && make install && \
    cd .. && rm cmake-${CMAKE_VERSION}.tar.gz && rm -r cmake-${CMAKE_VERSION}

RUN pip install numpy

# Install faiss
RUN git clone -b v1.7.1 --depth 1 https://github.com/facebookresearch/faiss.git && \
    cd faiss && \
    cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON && \
    make -C build -j faiss && \
    make -C build -j swigfaiss && \
    (cd build/faiss/python && python setup.py install)


WORKDIR /work
COPY tutorial.py .
CMD python tutorial.py
