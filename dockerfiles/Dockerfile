# src: https://github.com/NVIDIA/retinanet-examples/blob/main/Dockerfile.deepstream
FROM nvcr.io/nvidia/pytorch:21.12-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.4.0
ARG DEEPSTREAM_SDK_VERSION=6.1.1
ARG DEEPSTREAM_VERSION=6.1

WORKDIR /opt/
RUN apt-get update && apt-get install -y libssl1.1 \
    libgstreamer1.0-0 gstreamer1.0-tools \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstrtspserver-1.0-0 libjansson4 ffmpeg \
    libjson-glib-1.0 libgles2-mesa libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libjson-glib-dev \
    libgstrtspserver-1.0-dev

# Install pkg config to find ffmpeg
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

# For decode
WORKDIR /usr/lib/x86_64-linux-gnu
RUN ln -sf libnvcuvid.so.1 libnvcuvid.so
# RUN apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev

WORKDIR /opt/
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \ 
    cd nv-codec-headers && make install && cd .. && \ 
    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/ && \ 
    apt-get update && apt-get install -y build-essential yasm libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev && \ 
    cd ffmpeg && \
    ./configure --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared --disable-x86asm && \ 
    make -j$(nproc) && \
    make install 

# To install Kafka
RUN git clone https://github.com/edenhill/librdkafka.git /librdkafka && \
    cd /librdkafka && ./configure && make -j && make -j install && \
    mkdir -p /opt/nvidia/deepstream/deepstream-6.0/lib && \
    cp /usr/local/lib/librdkafka* /opt/nvidia/deepstream/deepstream-6.0/lib && \
    rm -rf /librdkafka

# To install Deepstream sdk and samples
COPY ./deepstream_sdk_v${DEEPSTREAM_SDK_VERSION}_x86_64.tbz2 /
WORKDIR /   
RUN tar -xvf deepstream_sdk_v${DEEPSTREAM_SDK_VERSION}_x86_64.tbz2
WORKDIR /opt/nvidia/deepstream/deepstream-${DEEPSTREAM_VERSION}
RUN ./install.sh
# conofig files + sample apps
RUN chmod u+x ./sources/tools/nvds_logger/setup_nvds_logger.sh


# For convert to tensorrt
RUN pip install onnxruntime-gpu

# For other image operation, use OpenCV
WORKDIR /opt/
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip
RUN  wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip
WORKDIR /opt/
RUN unzip opencv.zip && unzip opencv_contrib.zip
WORKDIR /opt/opencv-${OPENCV_VERSION}/build/
RUN cmake \
      -D BUILD_opencv_java=OFF \
      -D WITH_FFMPEG=OFF \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_PTX=7.5 \
      -D WITH_NVCUVID=ON \
      -D WITH_CUFFT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_QT=ON \
      -D WITH_IPP=ON \
      -D WITH_TBB=OFF \
      -D WITH_EIGEN=ON \
      -D BUILD_opencv_java=OFF \
      -D BUILD_opencv_python=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_aruco=ON \
      -D BUILD_opencv_bgsegm=ON \
      -D BUILD_opencv_bioinspired=OFF \
      -D BUILD_opencv_ccalib=ON \
      -D BUILD_opencv_datasets=ON \
      -D BUILD_opencv_dnn_objdetect=ON \
      -D BUILD_opencv_dpm=OFF \
      -D BUILD_opencv_fuzzy=OFF \
      -D BUILD_opencv_hfs=OFF \
      -D BUILD_opencv_java_bindings_generator=OFF \
      -D BUILD_opencv_js=OFF \
      -D BUILD_opencv_img_hash=ON \
      -D BUILD_opencv_line_descriptor=ON \
      -D BUILD_opencv_optflow=OFF \
      -D BUILD_opencv_phase_unwrapping=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_python_bindings_generator=OFF \
      -D BUILD_opencv_reg=ON \
      -D BUILD_opencv_rgbd=ON \
      -D BUILD_opencv_saliency=OFF \
      -D BUILD_opencv_shape=OFF \
      -D BUILD_opencv_stereo=ON \
      -D BUILD_opencv_stitching=OFF \
      -D BUILD_opencv_structured_light=ON \
      -D BUILD_opencv_superres=OFF \
      -D BUILD_opencv_surface_matching=ON \
      -D BUILD_opencv_ts=OFF \
      -D BUILD_opencv_cudacodec=ON \
      -D BUILD_opencv_xobjdetect=OFF \
      -D BUILD_opencv_xphoto=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_QT=OFF \
      -D WITH_GTK=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      -D OpenCL_LIBRARY=/usr/local/cuda-11.5/lib64/libOpenCL.so \
      -D OpenCL_INCLUDE_DIR=/usr/local/cuda-11.5/include/ \
      -D CMAKE_BUILD_TYPE=RELEASE \
        .. &&\
    make -j$(nproc) && \
    make install && \
    ldconfig

# For fast search feature: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
WORKDIR /opt/
RUN git clone https://github.com/facebookresearch/faiss.git -b v1.7.3
WORKDIR /opt/faiss/
RUN apt-get update && apt-get install -y swig
RUN cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_OPT_LEVEL=avx2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT=/usr/local/cuda-11.5 \
    -DBLA_VENDOR=OpenBLAS

RUN make -C build -j$(nproc) faiss
RUN make -C build -j$(nproc) swigfaiss
RUN cd ./build/faiss/python && python setup.py install 
RUN make -C build install
RUN cd build/faiss/python && python setup.py build 

WORKDIR /workspace/

RUN apt-get update && apt-get install -y libgtk2.0-dev

RUN python --version

WORKDIR /opt/
RUN git clone -b v2021.1 https://github.com/inducer/pycuda.git && cd pycuda && \
	git submodule update --init --recursive && python3.8 configure.py \
			--cuda-root=/usr/local/cuda \
			--cuda-enable-gl && \
	make install

# ffcv
RUN apt-get update && apt-get install -y libturbojpeg libturbojpeg-dev 
WORKDIR /opt/
RUN git clone https://github.com/libffcv/ffcv.git
WORKDIR /opt/ffcv/
RUN python setup.py develop --user

# YOLOX 
WORKDIR /opt/
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR /opt/YOLOX/
RUN python setup.py develop


WORKDIR /workspace/
# ARG USERNAME=thanhnv
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# # Create the user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# # ********************************************************
# # * Anything else you want to do like clean up goes here *
# # ********************************************************

# # [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
