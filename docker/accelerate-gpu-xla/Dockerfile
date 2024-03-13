FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1
RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/huggingface/accelerate#egg=accelerate[test_prod,test_integrations] \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Activate the virtualenv
CMD ["/bin/bash"]