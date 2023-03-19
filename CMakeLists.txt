cmake_minimum_required(VERSION 3.20)
project(tensorRT)

#set self initial value when it has not default value, it does not work when variables have default value
#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

#'set' will work whether variables have default value
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(TRT_DIR "F:/TensorRT_master/TensorRT/TensorRT-8.5.3.1")
set(TRT_INCLUDE_DIRS ${TRT_DIR}/include)
set(TRT_LIB_DIRS ${TRT_DIR}/lib)
set(TRT_COMMON_DIRS ${TRT_DIR}/samples/common)
set(TRT_COMMON_WINDWOS_DIRS ${TRT_DIR}/samples/common/windows)
set(CMAKE_CXX_STANDARD 11)


set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64")
set(CUDA_LIB_DIRS "${CUDA_DIR}/lib/x64/")
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_DIR})
set(CMAKE_CUDA_COMPILER ${CUDA_DIR}/bin/nvcc.exe)
set(OpenCV_DIR "E:/opencv-4.5.4/opencv/build/x64/vc16")
#if(WIN32)
#enable_language(CUDA)
#endif(WIN32)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_INCLUDE_DIRS})
include_directories(${TRT_COMMON_DIRS})
include_directories(${TRT_COMMON__WINDWOS_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


FILE(GLOB SRCS "./src/*")
add_executable(${PROJECT_NAME} syn_main.cpp ${SRCS})

target_link_directories(${PROJECT_NAME} PUBLIC ${TRT_LIB_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_LIB_DIRS})
#注意动态库链接顺序如果动态库包含自己的src包
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser nvinfer_plugin nvparsers)
target_link_libraries(${PROJECT_NAME} cudart cudnn cublas)

#add_definitions(-O2 -pthread)

message(STATUS "cuda_include_path:${CUDA_INCLUDE_DIRS}")
message(STATUS "cuda_libraries:${CUDA_LIBRARIES}")
message(STATUS "opencv_libraries:${OpenCV_LIBS}")
message(STATUS "opencv_include_dirs:${OpenCV_INCLUDE_DIRS}")
message(STATUS "project_source_dir:${PROJECT_SOURCE_DIR}")
message(STATUS "tensorRT_version:${TRT_DIR}")
#message(STATUS "cuda_cmake_link_path:${CUDNN_LIBRARIES}")

