# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size
# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.6 
# Install apt libs
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
# Activate usergroup
RUN useradd -ms /bin/bash user
USER user
ENV PATH /home/user/.local/bin$:PATH
# Activate the conda env and install torch + accelerate
RUN source activate accelerate && \
    python3 -m pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cu113 \
    git+https://github.com/huggingface/accelerate#egg=accelerate[dev]

# Stage 2
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
COPY --from=compile-image /home/user /home/user
ENV PATH /opt/conda/bin:$PATH

RUN echo "source activate accelerate" >> /.bashrc

RUN useradd -ms /bin/bash user
USER user
# Activate the virtualenv
CMD ["/bin/bash"]