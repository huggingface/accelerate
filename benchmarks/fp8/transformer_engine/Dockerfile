ARG BASE_YEAR=25
ARG BASE_MONTH=03

FROM nvcr.io/nvidia/pytorch:${BASE_YEAR}.${BASE_MONTH}-py3-igpu

RUN pip install transformers evaluate datasets
RUN git clone https://github.com/huggingface/accelerate.git

RUN cd accelerate && \
    pip install -e . && \
    cd benchmarks/fp8

RUN /bin/bash


