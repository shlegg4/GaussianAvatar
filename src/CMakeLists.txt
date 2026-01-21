cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Enable C++ and CUDA support
project(smpl_lbs_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- CONFIGURATION START ---
# SET THIS: Path to your TensorRT install folder
set(TENSORRT_ROOT "D:/TensorRT-8.6.1.6")
# SET THIS: Path to your CUDA install folder
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
# SET THIS: Name of your ONNX model file (must be in the same folder as main.cpp)
set(ONNX_MODEL_FILE "D:/Dissertation/Compiled_GaussianAvatar/cliff_hr48_static.onnx")
# --- CONFIGURATION END ---

# ==========================================
# DEPENDENCY 1: LibTorch
# ==========================================
list(APPEND CMAKE_PREFIX_PATH "D:/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ==========================================
# DEPENDENCY 2: OpenCV & TensorRT
# ==========================================
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)

if(NOT EXISTS "${TENSORRT_ROOT}")
    message(FATAL_ERROR "TENSORRT_ROOT not found at ${TENSORRT_ROOT}. Please update the path in CMakeLists.txt")
endif()
if(NOT EXISTS "${CUDA_ROOT}")
    message(FATAL_ERROR "CUDA_ROOT not found at ${CUDA_ROOT}. Please update the path in CMakeLists.txt")
endif()
include_directories("${TENSORRT_ROOT}/include")
include_directories("${CUDA_ROOT}/include")
link_directories("${TENSORRT_ROOT}/lib")
if(MSVC)
    link_directories("${CUDA_ROOT}/lib/x64")
else()
    link_directories("${CUDA_ROOT}/lib64")
endif()

# ==========================================
# SUBPROJECTS
# ==========================================
# This looks into 'third_party/diff-gaussian-rasterization' and builds the library defined there
add_subdirectory(third_party/diff-gaussian-rasterization)

# ==========================================
# EXECUTABLE 1: smpl_test (LibTorch)
# ==========================================
add_executable(smpl_test main.cpp)

target_link_libraries(smpl_test 
    PRIVATE 
    "${TORCH_LIBRARIES}" 
    GaussianRasterizer
)
target_compile_definitions(smpl_test PRIVATE NOMINMAX)

# ==========================================
# EXECUTABLE 2: hmr_inference (ONNX/OpenCV)
# ==========================================
add_executable(hmr_inference
    hmr_inference.cpp
    utils/HmrInferenceUtils.cpp
)

if(MSVC)
    target_link_libraries(hmr_inference 
        PRIVATE 
        ${OpenCV_LIBS}
        nvinfer.lib
        nvonnxparser.lib
        cudart.lib
        ${TORCH_LIBRARIES}
    )
    target_compile_definitions(hmr_inference PRIVATE NOMINMAX)
    
    # Post-Build: Copy TensorRT and CUDA runtime DLLs next to the exe
    add_custom_command(TARGET hmr_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/lib/nvinfer.dll"
        $<TARGET_FILE_DIR:hmr_inference>
    )
    add_custom_command(TARGET hmr_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/lib/nvonnxparser.dll"
        $<TARGET_FILE_DIR:hmr_inference>
    )
    add_custom_command(TARGET hmr_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/lib/nvinfer_builder_resource.dll"
        $<TARGET_FILE_DIR:hmr_inference>
    )
    add_custom_command(TARGET hmr_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TENSORRT_ROOT}/lib/nvinfer_plugin.dll"
        $<TARGET_FILE_DIR:hmr_inference>
    )
    add_custom_command(TARGET hmr_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDA_ROOT}/bin/cudart64_12.dll"
        $<TARGET_FILE_DIR:hmr_inference>
    )
else()
    target_link_libraries(hmr_inference 
        PRIVATE 
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        cudart
        ${TORCH_LIBRARIES}
    )
endif()

# ==========================================
# FILE COPYING
# ==========================================

# 1. Copy smpl_data.pt (for smpl_test)
if(EXISTS "${CMAKE_SOURCE_DIR}/smpl_data.pt")
    file(COPY ${CMAKE_SOURCE_DIR}/smpl_data.pt DESTINATION ${CMAKE_BINARY_DIR})
endif()

# 2. Copy the ONNX model (for hmr_inference)
if(EXISTS "${CMAKE_SOURCE_DIR}/${ONNX_MODEL_FILE}")
    message(STATUS "Copying ${ONNX_MODEL_FILE} to build directory...")
    file(COPY ${CMAKE_SOURCE_DIR}/${ONNX_MODEL_FILE} DESTINATION ${CMAKE_BINARY_DIR})
else()
    message(WARNING "Could not find ${ONNX_MODEL_FILE} in source directory. You will need to copy it manually.")
endif()
