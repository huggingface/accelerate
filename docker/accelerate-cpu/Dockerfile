
FROM ubuntu:18.04 as compile-image
RUN apt update
RUN apt-get install -y --no-install-recommends bash \
    build-essential \
    gcc \
    python3.6-dev \
    python3-pip \
    python3-venv 

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Make sure we use the virtualenv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /workspace
COPY . accelerate/
RUN python3 -m pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu
    
RUN cd accelerate/ && \
    python3 -m pip install --no-cache-dir .

FROM ubuntu:18.04 AS build-image
COPY --from=compile-image /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]