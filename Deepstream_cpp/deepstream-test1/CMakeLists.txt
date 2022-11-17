cmake_minimum_required(VERSION 3.2)
project(deepstream)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -pthread")

add_executable(deepstream_test1 deepstream_test1.cpp)

include_directories(deepstream_test1
  ../app-common/includes/
  ../includes/
  /usr/local/cuda-11.4/include
  /usr/include/gstreamer-1.0
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

link_directories(
  /usr/local/cuda-11.4/lib64/
  /opt/nvidia/deepstream/deepstream-6.0/lib/
  /usr/local/cuda/lib64/
)

target_link_libraries(deepstream_test1
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
)