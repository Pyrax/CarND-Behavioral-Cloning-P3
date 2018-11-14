ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-runtime

MAINTAINER Pyrax

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# -- Install tools to build python manually and some OpenCV dependencies:
RUN apt update && apt upgrade -y && apt install -y --no-install-recommends \
    build-essential \
    checkinstall \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    zlib1g-dev \
    openssl \
    libffi-dev \
    python3-dev \
    python3-setuptools \
    curl \
    ffmpeg \
    libpng-dev \
    libsm-dev \
    pkg-config \
    libgtk2.0-0 \
    && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/Python36 && \
    cd /tmp/Python36

RUN curl --silent https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz | \
    tar xJ -C /tmp/Python36 && \
    cd /tmp/Python36/Python-3.6.0 && \
    ./configure --enable-optimizations && \
    make altinstall

RUN curl --silent https://bootstrap.pypa.io/get-pip.py | python3.6

# Backwards compatility.
RUN rm -rf /usr/bin/python3 && \
    mv /tmp/Python36/Python-3.6.0/python /usr/bin/python3.6 && \
    ln /usr/bin/python3.6 /usr/bin/python3

RUN pip3 install pipenv

# -- Install application:
RUN set -ex && mkdir /src
WORKDIR /src

# Dependencies with pipfiles
COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN set -ex && pipenv install --deploy --system

# Jupyter
EXPOSE 8888
# Flask server
EXPOSE 4567

CMD pipenv run jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 --NotebookApp.token=
