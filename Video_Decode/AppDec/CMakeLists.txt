cmake_minimum_required(VERSION 3.2)
project(deepstream)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -pthread")

find_library(AVCODEC_LIBRARY avcodec)

message(STATUS "AVCODEC_LIBRARY    version: ${AVCODEC_LIBRARY}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

add_executable(AppDec AppDec.cpp)

include_directories(AppDec
  ../app-common/includes/
  ../includes/
  /usr/local/cuda-11.4/include
  /usr/include/gstreamer-1.0
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
  /usr/include/x86_64-linux-gnu/libavcodec/
)

link_directories(
  /usr/local/cuda-11.4/lib64/
  /opt/nvidia/deepstream/deepstream-6.1/lib/
  /usr/local/cuda/lib64/
  /usr/lib/x86_64-linux-gnu/
  /usr/local/ffmpeg/lib/
  ../libs/
)

target_link_libraries(AppDec
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
  /usr/lib/x86_64-linux-gnu/libavutil.so
  /usr/local/ffmpeg/lib/libavcodec.so
  /usr/local/ffmpeg/lib/libavdevice.so
  /usr/local/ffmpeg/lib/libavfilter.so
  /usr/local/ffmpeg/lib/libavformat.so
  /usr/local/ffmpeg/lib/libavutil.so
  /usr/local/ffmpeg/lib/libswresample.so
  /usr/local/ffmpeg/lib/libswscale.so
  ${AVCODEC_LIBRARY}
)

