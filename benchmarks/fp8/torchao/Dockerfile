FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install transformers evaluate datasets
RUN git clone https://github.com/huggingface/accelerate.git

RUN cd accelerate && \
    pip install -e . && \
    cd benchmarks/fp8

RUN /bin/bash


