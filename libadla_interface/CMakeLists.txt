cmake_minimum_required(VERSION 3.5.1)
SET(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

PROJECT(AML_NN_SDK)
SET(INCLUDE_PATH ../nn_sdk)
include_directories(${INCLUDE_PATH}/include ./include)

SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
SET(LINK_DIR ../nn_sdk/lib/lib64  ./)

link_directories(${LINK_DIR})
aux_source_directory(. SRC_LIST)
link_libraries(nnsdk)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
ADD_LIBRARY(adla_interface SHARED ${SRC_LIST})
target_link_libraries(adla_interface -lpthread -lm -ldl)

