cmake_minimum_required(VERSION 3.2)
project(deepstream)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -pthread")

find_package(PkgConfig) 
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
pkg_search_module(gstreamer-sdp REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
pkg_search_module(gstreamer-video REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.4)


add_executable(deepstream_test1 
  deepstream_test1.cpp
  ClassExample.h
  ClassExample.cpp
)

include_directories(deepstream_test1
  ../app-common/includes/
  ../logger/
  ../includes/
  /usr/local/cuda-11.5/include
  /usr/include/gstreamer-1.0
  /usr/include/glib-2.0
  /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

link_directories(
  /usr/local/cuda-11.5/lib64/
  /opt/nvidia/deepstream/deepstream-6.0/lib/
  /opt/nvidia/deepstream/deepstream-6.0/lib/gst-plugins/
  /opt/nvidia/deepstream/deepstream-6.0/lib/cvcore_libs/
  /opt/nvidia/deepstream/deepstream-6.0/lib/libv4l/plugins/
  /usr/local/cuda/lib64/
  )
  
target_include_directories(deepstream_test1 
PRIVATE ${GST_INCLUDE_DIRS})

target_link_libraries(deepstream_test1
  ${CUDA_LIBRARIES}
  /usr/local/cuda/lib64/libcudart.so
  ${GST_LIBRARIES}
  PkgConfig::gstreamer
  PkgConfig::gstreamer-sdp
  PkgConfig::gstreamer-app
  PkgConfig::gstreamer-video
  /opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_osd.so
  /opt/nvidia/deepstream/deepstream-6.0/lib/libnvdsgst_meta.so
  /opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_meta.so
  /opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_infer_server.so
  /opt/nvidia/deepstream/deepstream-6.0/lib/gst-plugins/libnvdsgst_nvblender.so
  
)

add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
add_definitions( -D LOGGER_PREFIX_LEVEL=2 )
add_definitions( -D LOGGER_ENABLE_COLORS=1 )
add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
