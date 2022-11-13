# Replace with required container type e.g. base, devel etc in the following line
FROM nvcr.io/nvidia/deepstream:6.0.1-devel

ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80

RUN cat /etc/lsb-release
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/

RUN apt-get update -qq  && apt-get install -y autoconf \
    automake \
    build-essential \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev

RUN apt install -y git python-dev python3 python3-pip python3.6-dev python3.8-dev cmake g++ build-essential \
    libglib2.0-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

RUN apt-get install -y libavdevice-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev libavdevice-dev

RUN apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev

WORKDIR /opt/nvidia/deepstream/deepstream/sources/
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
RUN git submodule update --init
RUN apt-get install -y apt-transport-https ca-certificates -y && update-ca-certificates
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/3rdparty/gst-python/
RUN ./autogen.sh && make && make install 
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/build/
RUN cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DDS_VERSION=6.0 .. && make


WORKDIR /workspace/

