# Builds CPU-only Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM ubuntu:18.04 as compile-image
RUN apt update
RUN apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    gcc \
    python3.6-dev \
    python3-pip \
    python3-venv 

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
FROM ubuntu:18.04 AS build-image
COPY --from=compile-image /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]