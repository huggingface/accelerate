FROM ghcr.io/azure/msamp

RUN pip install transformers evaluate datasets
RUN git clone https://github.com/huggingface/accelerate

RUN cd accelerate && \
    pip install -e . && \
    cd benchmarks/fp8

CMD ["bash"]


