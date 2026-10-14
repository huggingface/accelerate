ARG BASE_YEAR=26
ARG BASE_MONTH=09

FROM nvcr.io/nvidia/pytorch:${BASE_YEAR}.${BASE_MONTH}-py3

ARG TE_VERSION=2.20.2
RUN PIP_CONSTRAINT= MAX_JOBS=2 NVTE_BUILD_THREADS_PER_JOB=1 \
    pip install --no-cache-dir --no-build-isolation --upgrade "transformer_engine[pytorch,core_cu13]==${TE_VERSION}"

# The container pins Datasets 4.7, whose VideoReader import is incompatible with its newer TorchVision.
RUN PIP_CONSTRAINT= pip install --no-cache-dir transformers evaluate "datasets>=5.1.0" scikit-learn

WORKDIR /workspace/accelerate
COPY . .
RUN pip install --no-cache-dir -e .[deepspeed]

WORKDIR /workspace/accelerate/benchmarks/fp8/transformer_engine
CMD ["/bin/bash"]
