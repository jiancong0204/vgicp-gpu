cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")
set(CMAKE_CXX_STANDARD 11)

project(cuda_tutorial)

file(GLOB_RECURSE CUDA_SRCS "src/*.cu")
file(GLOB_RECURSE CPP_SRCS "src/*.cpp")

cmake_policy(SET CMP0104 NEW)

#cuda
enable_language("CUDA")
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS -arch=compute_75 --expt-relaxed-constexpr>)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    include
)

add_executable(${PROJECT_NAME} "main.cpp" ${CPP_SRCS} ${CUDA_SRCS})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${CUDA_NVCC_FLAGS})