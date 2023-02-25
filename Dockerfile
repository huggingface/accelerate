# Builds CPU/GPU docker image of PyTorch
# Uses multi-staged approach to reduce size

ARG BASE_IMAGE

# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image

ARG BASE_REPOSITORY
ARG PYTHON_VERSION
ARG PYTORCH_COMPUTE_PLATFORM

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Create our conda env
RUN conda create --name accelerate python=${PYTHON_VERSION} ipython jupyter pip

# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/accelerate/bin:$PATH

# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Activate the conda env and install torch + accelerate
RUN source activate accelerate && \
    python3 -m pip install --no-cache-dir git+${BASE_REPOSITORY} \
    --extra-index-url https://download.pytorch.org/whl/${PYTORCH_COMPUTE_PLATFORM}

# Stage 2
FROM ${BASE_IMAGE} AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

ARG INSTALL_DOCKER
ARG INSTALL_AWS_CLI
ARG INSTALL_DOCKER_COMPOSE_V1

# Install dependencies
RUN apt-get update && \
    apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists
    
RUN echo "source activate accelerate" >> ~/.profile

# Install docker to interact with host docker daemon
RUN if [ ! -z "$INSTALL_DOCKER" ]; then curl -fsSL https://get.docker.com -o get-docker.sh && \
    chmod +x get-docker.sh && ./get-docker.sh ; fi

# Install aws cli v2
RUN if [ ! -z "$INSTALL_AWS_CLI" ]; then curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install ; fi

# Install docker-compose v1
RUN if [ ! -z "$INSTALL_DOCKER_COMPOSE_V1" ]; then curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose ; fi

# Install docker compose v2
# (Note: not required for now, SageMaker local mode doesn't support it yet)
# RUN if [ ! -z "$INSTALL_DOCKER_COMPOSE_V1" ]; then curl -SL https://github.com/docker/compose/releases/download/v2.16.0/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose && \
#     chmod +x ~/.docker/cli-plugins/docker-compose ; fi

# Activate bash shell and conda env
CMD ["/bin/bash"]