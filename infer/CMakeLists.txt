cmake_minimum_required(VERSION 3.1)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS --std=c++11)
set(CMAKE_CXX_STANDARD 11)

cuda_add_executable(infer infer.cu)
target_link_libraries(infer ${OpenCV_LIBS} nvinfer nvparsers)

