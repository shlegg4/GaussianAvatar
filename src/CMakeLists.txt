cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Enable C++ and CUDA support
project(smpl_lbs_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(MSVC)
    # Only apply the /utf-8 flag to C++ files, NOT CUDA files
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    
    # If NVCC also needs it, it must be wrapped in -Xcompiler
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# --- CONFIGURATION START ---
# SET THIS: Path to your TensorRT install folder
set(TENSORRT_ROOT "C:/TensorRT-8.6.1.6")
# SET THIS: Path to your CUDA install folder
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
# SET THIS: Name of your ONNX model file
set(ONNX_MODEL_FILE "D:/Dissertation/Compiled_GaussianAvatar/cliff_hr48_static.onnx")
# SET THIS: Path to your ONNX Runtime install folder
set(ONNXRUNTIME_ROOT "C:/onnxruntime-win-x64-gpu-1.23.2" )

set(Open3D_DIR "C:/open3d/CMake")
# --- CONFIGURATION END ---

# ==========================================
# DEPENDENCY 1: LibTorch
# ==========================================
list(APPEND CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ==========================================
# DEPENDENCY 2: OpenCV & TensorRT
# ==========================================
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

option(ENABLE_OPEN3D_TSDF "Enable Open3D-based TSDF mesh extraction" ON)
if(ENABLE_OPEN3D_TSDF)
    find_package(Open3D QUIET)
    if(Open3D_FOUND)
        message(STATUS "Open3D found: ${Open3D_DIR}")
    else()
        message(WARNING "Open3D not found. TSDF mesh extraction will be disabled. "
                        "Set Open3D_DIR or CMAKE_PREFIX_PATH to your Open3D install.")
    endif()
endif()

option(ENABLE_VIEWER "Build GLFW viewer support" ON)
if(ENABLE_VIEWER)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG QUIET)
    message(STATUS "OpenGL found: ${OpenGL_FOUND}")
    message(STATUS "glfw3 found: ${glfw3_FOUND}")
    if(NOT glfw3_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4
        )
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
        message(STATUS "glfw3 provided via FetchContent")
    endif()
endif()

if(NOT EXISTS "${TENSORRT_ROOT}")
    message(FATAL_ERROR "TENSORRT_ROOT not found at ${TENSORRT_ROOT}. Please update the path in CMakeLists.txt")
endif()
if(NOT EXISTS "${CUDA_ROOT}")
    message(FATAL_ERROR "CUDA_ROOT not found at ${CUDA_ROOT}. Please update the path in CMakeLists.txt")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT not found at ${ONNXRUNTIME_ROOT}. Please update the path in CMakeLists.txt")
endif()
include_directories("${TENSORRT_ROOT}/include")
include_directories("${CUDA_ROOT}/include")
include_directories("${ONNXRUNTIME_ROOT}/include")
link_directories("${TENSORRT_ROOT}/lib")
link_directories("${ONNXRUNTIME_ROOT}/lib")
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
add_executable(smpl_test
    main.cpp
    utils/HmrMathHelpers.cpp
    utils/image/MaskUtils.cpp
    utils/image/TensorCvUtils.cpp
    utils/io/PathUtils.cpp
    utils/math/StatsUtils.cpp
    utils/render/RenderMathUtils.cpp
    utils/train/TrainImageSaver.cpp
    utils/train/TrainJsonl.cpp
    utils/train/GaussianDensification.cpp
    utils/train/ViewerExport.cpp
)

target_link_libraries(smpl_test 
    PRIVATE 
    "${TORCH_LIBRARIES}" 
    GaussianRasterizer
    ${OpenCV_LIBS}
)
if(ENABLE_OPEN3D_TSDF AND Open3D_FOUND)
    target_link_libraries(smpl_test PRIVATE Open3D::Open3D)
    target_compile_definitions(smpl_test PRIVATE GAUSS_HAS_OPEN3D=1)
else()
    target_compile_definitions(smpl_test PRIVATE GAUSS_HAS_OPEN3D=0)
endif()
target_include_directories(smpl_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(ENABLE_VIEWER)
    target_compile_definitions(smpl_test PRIVATE GAUSS_VIEWER)
    target_link_libraries(smpl_test PRIVATE OpenGL::GL glfw)
endif()
target_compile_definitions(smpl_test PRIVATE NOMINMAX)

# ==========================================
# EXECUTABLE 1.5: gaussian_viewer (OpenGL)
# ==========================================
if(ENABLE_VIEWER)
    add_executable(gaussian_viewer
        gaussian_viewer.cpp
        utils/viewer/ViewerMath.cpp
    )
    target_link_libraries(gaussian_viewer PRIVATE OpenGL::GL glfw "${TORCH_LIBRARIES}" GaussianRasterizer)
    target_include_directories(gaussian_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(gaussian_viewer PRIVATE NOMINMAX)

    add_executable(gaussian_viewer_stdout
        gaussian_viewer_stdout.cpp
        utils/viewer/ViewerMath.cpp
    )
    target_link_libraries(gaussian_viewer_stdout PRIVATE OpenGL::GL glfw "${TORCH_LIBRARIES}" GaussianRasterizer)
    target_include_directories(gaussian_viewer_stdout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(gaussian_viewer_stdout PRIVATE NOMINMAX)
endif()

# ==========================================
# EXECUTABLE 2: hmr_inference (ONNX/OpenCV)
# ==========================================
add_executable(hmr_inference
    hmr_inference.cpp
    utils/HmrInferenceUtils.cpp
    utils/HmrMathHelpers.cpp
    utils/HmrOverlayHelpers.cpp
    utils/SmplifyLite.cpp
    utils/TrtBuilder.cpp
    utils/RtmPoseDetector.cpp
    utils/YoloPersonDetector.cpp
    utils/ModNetMatte.cpp
)
target_include_directories(hmr_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_link_libraries(hmr_inference 
        PRIVATE 
        ${OpenCV_LIBS}
        nvinfer.lib
        nvonnxparser.lib
        cudart.lib
        onnxruntime.lib
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
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        add_custom_command(TARGET hmr_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:hmr_inference>
        )
    endif()
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
        add_custom_command(TARGET hmr_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:hmr_inference>
        )
    endif()
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
        add_custom_command(TARGET hmr_inference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:hmr_inference>
        )
    endif()
else()
    target_link_libraries(hmr_inference 
        PRIVATE 
        ${OpenCV_LIBS}
        nvinfer
        nvonnxparser
        cudart
        onnxruntime
        ${TORCH_LIBRARIES}
    )
endif()


if(ENABLE_OPEN3D_TSDF AND Open3D_FOUND)
    add_custom_command(TARGET smpl_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "C:/open3d/bin/Open3D.dll"  # Update this if your path is different
        $<TARGET_FILE_DIR:smpl_test>
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

