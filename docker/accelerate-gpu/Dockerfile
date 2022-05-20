# Builds GPU Docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 as compile-image
ENV PYTHON_VERSION=3.6

RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
# Install apt libs
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    find /opt/conda/ -follow -type f -name "*.a" -delete \
    find /opt/conda/ -follow -type f -name "*.js.map" -delete && \
    /opt/conda/bin/conda clean -afy && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH="/opt/conda/bin:$PATH"

# Setup conda venv
RUN conda create --name accelerate python=${PYTHON_VERSION} ipython pytorch jupyter pip -c pytorch
ENV PATH /opt/conda/envs/accelerate/bin:$PATH
RUN source activate accelerate && \
    git clone https://github.com/huggingface/accelerate && \
    cd accelerate/ && \
    python3 -m pip install --no-cache-dir .[dev]

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
RUN echo "conda activate accelerate" >> ~/.bashrc
# Activate the venv
CMD ["source", "~/.bashrc"]