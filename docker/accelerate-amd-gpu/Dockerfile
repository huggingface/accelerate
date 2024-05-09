FROM rocm/dev-ubuntu-20.04:5.6
# rocm/pytorch has no version with 2.1.0
ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH='2.1.0'
ARG TORCH_VISION='0.16.0'
ARG ROCM='5.6'

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev python3 python3-dev python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Make sure we use the virtualenv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    jupyter \
    torch==$PYTORCH \
    torchvision==$TORCH_VISION \
    git+https://github.com/huggingface/accelerate#egg=accelerate[testing,test_trackers] \
    --extra-index-url https://download.pytorch.org/whl/rocm$ROCM

# Remove nvml as it is not compatible with ROCm
RUN python3 -m pip uninstall py3nvml pynvml -y
CMD ["/bin/bash"]