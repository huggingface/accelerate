# Builds GPU Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as compile-image
ENV PYTHON_VERSION=3.6

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
# Install anaconda
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda init bash

# Set path to conda
ENV PATH="/opt/conda/bin:$PATH"

# Setup conda venv
RUN conda create --name accelerate python=${PYTHON_VERSION} ipython pytorch jupyter pip && \
    conda activate accelerate && \
    git clone https://github.com/huggingface/accelerate && \
    cd accelerate/ && \
    python3 -m pip install --no-cache-dir .[dev]

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
COPY --from=compile-image /root/.bashrc /root/.bashrc
RUN echo "conda activate accelerate" >> ~/.bashrc

# Make sure we use the virtualenv
ENV PATH="/opt/conda/bin:$PATH"
RUN ["source", "~/.bashrc"]