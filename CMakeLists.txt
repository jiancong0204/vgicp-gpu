cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 11)

project(fast_gicp)

message("Build Type: ${CMAKE_BUILD_TYPE}")
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp -lpthread -Werror")
else()
    set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -O0 -g -std=c++11 -lpthread -Werror")
endif()

cmake_policy(SET CMP0104 NEW)

#cuda
enable_language("CUDA")
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS -arch=compute_75 --expt-relaxed-constexpr>)

#------------------------------------------------------------------
#thirdparty
#------------------------------------------------------------------
set(protobuf_MODULE_COMPATIBLE
    ON
    CACHE BOOL "") # protobuf in thirdparty.cmake is using CONFIG mode
include(${CMAKE_CURRENT_LIST_DIR}/thirdparty/thirdparty.cmake)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    include
)

file(GLOB_RECURSE CUDA_SRCS "src/*.cu")
file(GLOB_RECURSE CPP_SRCS "src/*.cpp")
add_executable(${PROJECT_NAME} "main.cpp" ${CPP_SRCS} ${CUDA_SRCS})

target_link_libraries(${PROJECT_NAME} 
    ${GLOG_LIBRARY}
    ${gflags_libraries}
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS})
