# Builds CPU-only Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM python:3.7-slim as compile-image

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc

# Setup virtual environment for Docker
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Make sure we use the virtualenv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
WORKDIR /workspace
# Install specific CPU torch wheel to save on space
RUN python3 -m pip install --upgrade --no-cache-dir pip
RUN python3 -m pip install --no-cache-dir \
    jupyter \
    git+https://github.com/huggingface/accelerate#egg=accelerate[testing,test_trackers] \
    --extra-index-url https://download.pytorch.org/whl/cpu
    
# Stage 2
FROM python:3.7-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv
RUN useradd -ms /bin/bash user
USER user

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]