# Builds GPU Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as compile-image

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt-get install -y --no-install-recommends \
    python3.6-dev \
    python3-pip \
    python3-setuptools

# Setup virtual environment for Docker
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Make sure we use the virtualenv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /workspace
COPY . accelerate/
# Install specific CPU torch wheel to save on space
RUN python3 -m pip install --no-cache-dir \
    jupyter \
    torch --extra-index-url https://download.pytorch.org/whl/cpu
    
RUN cd accelerate/ && \
    python3 -m pip install --no-cache-dir .

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]