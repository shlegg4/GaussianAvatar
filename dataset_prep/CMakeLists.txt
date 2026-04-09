cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(dataset_prep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DATASET_PREP_ENABLE_SMPL "Enable SMPL parameter fitting support" ON)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation used for MODNet")
set(Torch_ROOT "C:/libtorch" CACHE PATH "Path to a LibTorch installation used for SMPL fitting")

# Find OpenCV (vcpkg intercepts this if the vcpkg toolchain is used)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

set(DATASET_PREP_SOURCES
    DatasetPrepMain.cpp
    ingestion/CameraCalibrationLoader.cpp
    ingestion/MocapPoseParser.cpp
    ingestion/VideoSynchronizer.cpp
    processing/BackgroundExtractor.cpp
    processing/CliffEstimator.cpp
    export/DatasetExporter.cpp
)

add_executable(dataset_prep ${DATASET_PREP_SOURCES})

target_include_directories(dataset_prep
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(dataset_prep PRIVATE ${OpenCV_LIBS})
target_compile_definitions(dataset_prep PRIVATE NOMINMAX)

if(MSVC)
    target_compile_definitions(dataset_prep PRIVATE CV_IGNORE_DEBUG_BUILD_GUARD=1)

    # ============================================================================
    # VCPKG Runtime DLL Copying for OpenCV & FFmpeg
    # ============================================================================
    if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake")
        # _VCPKG_INSTALLED_DIR and VCPKG_TARGET_TRIPLET are provided by vcpkg.cmake
        set(VCPKG_BIN_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
        
        file(GLOB OPENCV_RUNTIME_DLLS
            # Core OpenCV
            "${VCPKG_BIN_DIR}/opencv_core*.dll"
            "${VCPKG_BIN_DIR}/opencv_imgproc*.dll"
            "${VCPKG_BIN_DIR}/opencv_imgcodecs*.dll"
            "${VCPKG_BIN_DIR}/opencv_videoio*.dll"
            "${VCPKG_BIN_DIR}/opencv_calib3d*.dll"
            
            # FFmpeg Codecs (Critical for videoio backend)
            "${VCPKG_BIN_DIR}/avformat*.dll"
            "${VCPKG_BIN_DIR}/avcodec*.dll"
            "${VCPKG_BIN_DIR}/avutil*.dll"
            "${VCPKG_BIN_DIR}/swscale*.dll"
            "${VCPKG_BIN_DIR}/swresample*.dll"
            
            # Image Codecs
            "${VCPKG_BIN_DIR}/libpng*.dll"
            "${VCPKG_BIN_DIR}/jpeg*.dll"
            "${VCPKG_BIN_DIR}/tiff*.dll"
            "${VCPKG_BIN_DIR}/zlib*.dll"

            # ONNX Runtime
            "${VCPKG_BIN_DIR}/onnxruntime*.dll"
        )
        
        if(OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET dataset_prep POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:dataset_prep>
                COMMENT "Copying OpenCV, FFmpeg, and Image Codec DLLs from vcpkg to output directory"
            )
        endif()
    endif()
endif()

if(DATASET_PREP_ENABLE_SMPL)
    list(APPEND CMAKE_PREFIX_PATH "${Torch_ROOT}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    target_sources(dataset_prep PRIVATE
        ../src/utils/HmrMathHelpers.cpp
        ../src/utils/RtmPoseDetector.cpp
        ../src/utils/SmplifyLite.cpp
    )
    target_link_libraries(dataset_prep PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(dataset_prep PRIVATE DATASET_PREP_HAS_SMPL=1)

    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        if(TORCH_DLLS)
            add_custom_command(TARGET dataset_prep POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:dataset_prep>
                COMMENT "Copying LibTorch DLLs to output directory"
            )
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../src/smpl_data.pt")
        add_custom_command(TARGET dataset_prep POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/../src/smpl_data.pt"
            $<TARGET_FILE_DIR:dataset_prep>
        )
    endif()
else()
    target_compile_definitions(dataset_prep PRIVATE DATASET_PREP_HAS_SMPL=0)
endif()
   
find_package(onnxruntime CONFIG REQUIRED) 
target_sources(dataset_prep PRIVATE utils/ModNetMatte.cpp) 
target_link_libraries(dataset_prep PRIVATE onnxruntime::onnxruntime)


set_property(TARGET dataset_prep PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
