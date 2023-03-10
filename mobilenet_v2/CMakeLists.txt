cmake_minimum_required(VERSION 3.5.1)
SET(CMAKE_SYSTEM_NAME Linux)

PROJECT(AML_NN_SDK)
SET(INCLUDE_PATH ../nn_sdk)
include_directories(${INCLUDE_PATH}/include ./include)

SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
SET(LINK_DIR ../nn_sdk/lib/lib64  ./)

link_directories(${LINK_DIR})
aux_source_directory(. SRC_LIST)
link_libraries(nnsdk libjpeg.a )

ADD_EXECUTABLE(mobilenetv2 ${SRC_LIST})
target_link_libraries(mobilenetv2 -lpthread -lm -ldl)

