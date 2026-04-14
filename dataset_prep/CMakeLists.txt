cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(dataset_prep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DATASET_PREP_ENABLE_SMPL "Enable SMPL parameter fitting support" ON)
if(WIN32)
    set(_dataset_prep_default_download_onnxruntime_gpu ON)
else()
    set(_dataset_prep_default_download_onnxruntime_gpu OFF)
endif()
option(DATASET_PREP_DOWNLOAD_ONNXRUNTIME_GPU "Download and use the official prebuilt Windows GPU ONNX Runtime package" ${_dataset_prep_default_download_onnxruntime_gpu})
unset(_dataset_prep_default_download_onnxruntime_gpu)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional path to an ONNX Runtime installation. Leave empty to use the downloaded GPU package on Windows.")
set(DATASET_PREP_ONNXRUNTIME_VERSION "1.23.2" CACHE STRING "ONNX Runtime version used by dataset_prep")
set(DATASET_PREP_ONNXRUNTIME_PACKAGE_NAME "onnxruntime-win-x64-gpu-${DATASET_PREP_ONNXRUNTIME_VERSION}")
set(DATASET_PREP_ONNXRUNTIME_DOWNLOAD_URL "https://github.com/microsoft/onnxruntime/releases/download/v${DATASET_PREP_ONNXRUNTIME_VERSION}/${DATASET_PREP_ONNXRUNTIME_PACKAGE_NAME}.zip")
set(Torch_ROOT "C:/libtorch" CACHE PATH "Path to a LibTorch installation used for SMPL fitting")

function(dataset_prep_import_prebuilt_onnxruntime out_root out_runtime_dir)
    if(NOT WIN32)
        message(FATAL_ERROR "DATASET_PREP_DOWNLOAD_ONNXRUNTIME_GPU currently supports Windows only. Set ONNXRUNTIME_ROOT or turn the option OFF on other platforms.")
    endif()

    if(ONNXRUNTIME_ROOT)
        set(_ort_root "${ONNXRUNTIME_ROOT}")
    else()
        set(_ort_archive_dir "${CMAKE_BINARY_DIR}/downloads")
        set(_ort_extract_dir "${CMAKE_BINARY_DIR}/third_party")
        set(_ort_archive_path "${_ort_archive_dir}/${DATASET_PREP_ONNXRUNTIME_PACKAGE_NAME}.zip")
        set(_ort_root "${_ort_extract_dir}/${DATASET_PREP_ONNXRUNTIME_PACKAGE_NAME}")

        file(MAKE_DIRECTORY "${_ort_archive_dir}")
        file(MAKE_DIRECTORY "${_ort_extract_dir}")

        if(NOT EXISTS "${_ort_archive_path}")
            message(STATUS "Downloading ONNX Runtime GPU ${DATASET_PREP_ONNXRUNTIME_VERSION} from ${DATASET_PREP_ONNXRUNTIME_DOWNLOAD_URL}")
            file(DOWNLOAD
                "${DATASET_PREP_ONNXRUNTIME_DOWNLOAD_URL}"
                "${_ort_archive_path}"
                STATUS _ort_download_status
                SHOW_PROGRESS
                TLS_VERIFY ON
            )
            list(GET _ort_download_status 0 _ort_download_code)
            list(GET _ort_download_status 1 _ort_download_message)
            if(NOT _ort_download_code EQUAL 0)
                message(FATAL_ERROR "Failed to download ONNX Runtime GPU archive: [${_ort_download_code}] ${_ort_download_message}")
            endif()
        endif()

        if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
            message(STATUS "Extracting ${DATASET_PREP_ONNXRUNTIME_PACKAGE_NAME}.zip to ${_ort_extract_dir}")
            file(ARCHIVE_EXTRACT
                INPUT "${_ort_archive_path}"
                DESTINATION "${_ort_extract_dir}"
            )
        endif()
    endif()

    if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime headers were not found under ${_ort_root}/include")
    endif()

    if(NOT EXISTS "${_ort_root}/lib/onnxruntime.lib")
        message(FATAL_ERROR "ONNX Runtime import library was not found at ${_ort_root}/lib/onnxruntime.lib")
    endif()

    if(NOT EXISTS "${_ort_root}/lib/onnxruntime.dll")
        message(FATAL_ERROR "ONNX Runtime DLL was not found at ${_ort_root}/lib/onnxruntime.dll")
    endif()

    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_ort_root}/include"
            IMPORTED_IMPLIB "${_ort_root}/lib/onnxruntime.lib"
            IMPORTED_LOCATION "${_ort_root}/lib/onnxruntime.dll"
        )

        foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_cfg})
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                "IMPORTED_IMPLIB_${_cfg}" "${_ort_root}/lib/onnxruntime.lib"
                "IMPORTED_LOCATION_${_cfg}" "${_ort_root}/lib/onnxruntime.dll"
            )
        endforeach()
    endif()

    message(STATUS "Using ONNX Runtime from ${_ort_root}")
    set(${out_root} "${_ort_root}" PARENT_SCOPE)
    set(${out_runtime_dir} "${_ort_root}/lib" PARENT_SCOPE)
endfunction()

# Find OpenCV (vcpkg intercepts this if the vcpkg toolchain is used)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d dnn viz)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

set(DATASET_PREP_ONNXRUNTIME_ROOT "")
set(DATASET_PREP_ONNXRUNTIME_RUNTIME_DIR "")

if(ONNXRUNTIME_ROOT OR DATASET_PREP_DOWNLOAD_ONNXRUNTIME_GPU)
    dataset_prep_import_prebuilt_onnxruntime(
        DATASET_PREP_ONNXRUNTIME_ROOT
        DATASET_PREP_ONNXRUNTIME_RUNTIME_DIR
    )
else()
    find_package(onnxruntime CONFIG REQUIRED)
    if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg.cmake")
        set(DATASET_PREP_ONNXRUNTIME_RUNTIME_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
    endif()
endif()

set(DATASET_PREP_SOURCES
    DatasetPrepMain.cpp
    ingestion/CameraCalibrationLoader.cpp
    ingestion/MocapPoseParser.cpp
    ingestion/VideoSynchronizer.cpp
    processing/BackgroundExtractor.cpp
    processing/SmplxOptimizer.cpp
    export/DatasetExporter.cpp
    utils/CommandLine.cpp
    utils/GeometryUtils.cpp
    utils/ImageUtils.cpp
    utils/ModNetMatte.cpp
    ../src/utils/RtmPoseDetector.cpp
    ../src/utils/YoloPersonDetector.cpp
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

add_executable(dataset_novel_view_render
    NovelViewRenderMain.cpp
    ../src/utils/train/TrainJsonl.cpp
)

target_include_directories(dataset_novel_view_render
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(dataset_novel_view_render PRIVATE ${OpenCV_LIBS})
target_compile_definitions(dataset_novel_view_render PRIVATE NOMINMAX)

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
            "${VCPKG_BIN_DIR}/opencv_dnn*.dll"
            "${VCPKG_BIN_DIR}/opencv_viz*.dll"
            "${VCPKG_BIN_DIR}/vtk*.dll"
            
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
        )
        
        if(OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET dataset_prep POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:dataset_prep>
                COMMENT "Copying OpenCV, FFmpeg, and image codec DLLs from vcpkg to output directory"
            )
        endif()
    endif()

    if(DATASET_PREP_ONNXRUNTIME_RUNTIME_DIR)
        file(GLOB ONNXRUNTIME_RUNTIME_DLLS
            "${DATASET_PREP_ONNXRUNTIME_RUNTIME_DIR}/onnxruntime*.dll"
        )

        if(ONNXRUNTIME_RUNTIME_DLLS)
            add_custom_command(TARGET dataset_prep POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:dataset_prep>
                COMMENT "Copying ONNX Runtime DLLs to output directory"
            )
        endif()
    endif()
endif()

if(DATASET_PREP_ENABLE_SMPL)
    list(APPEND CMAKE_PREFIX_PATH "${Torch_ROOT}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

    target_link_libraries(dataset_prep PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(dataset_prep PRIVATE DATASET_PREP_HAS_SMPL=1)

    target_link_libraries(dataset_novel_view_render PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(dataset_novel_view_render PRIVATE DATASET_PREP_HAS_SMPL=1)

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
    target_compile_definitions(dataset_novel_view_render PRIVATE DATASET_PREP_HAS_SMPL=0)
endif()

target_link_libraries(dataset_prep PRIVATE onnxruntime::onnxruntime)
target_link_libraries(dataset_novel_view_render PRIVATE onnxruntime::onnxruntime)


set_property(TARGET dataset_prep PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_property(TARGET dataset_novel_view_render PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
