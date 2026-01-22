#include "TrtBuilder.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <NvOnnxParser.h>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

void CheckCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << msg << ": " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

int64_t Volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        v *= dims.d[i];
    }
    return v;
}

TrtUniquePtr<nvinfer1::ICudaEngine> BuildEngineFromOnnx(const std::string& model_path,
                                                        TrtLogger& logger) {
    const std::filesystem::path engine_path = std::filesystem::path(model_path).string() + ".engine";
    static TrtUniquePtr<nvinfer1::IRuntime> runtime;
    if (std::filesystem::exists(engine_path)) {
        std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
        if (!engine_file) {
            throw std::runtime_error("Failed to open cached engine: " + engine_path.string());
        }
        const std::streamsize size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!engine_file.read(buffer.data(), size)) {
            throw std::runtime_error("Failed to read cached engine: " + engine_path.string());
        }

        if (!runtime) {
            runtime.reset(nvinfer1::createInferRuntime(logger));
        }
        if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime.");

        TrtUniquePtr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize cached engine: " + engine_path.string());
        }
        std::cout << "Loaded cached TensorRT engine: " << engine_path.string() << std::endl;
        return engine;
    }

    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("Failed to create TensorRT builder.");

    const uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    if (!network) throw std::runtime_error("Failed to create TensorRT network.");

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("Failed to create TensorRT ONNX parser.");

    if (!parser->parseFromFile(model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model: " + model_path);
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("Failed to create TensorRT builder config.");

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    TrtUniquePtr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config));
    if (!engine) throw std::runtime_error("Failed to build TensorRT engine.");

    TrtUniquePtr<nvinfer1::IHostMemory> serialized(engine->serialize());
    if (!serialized) throw std::runtime_error("Failed to serialize TensorRT engine.");

    std::ofstream out(engine_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open engine output: " + engine_path.string());
    }
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    if (!out) {
        throw std::runtime_error("Failed to write engine output: " + engine_path.string());
    }
    std::cout << "Saved TensorRT engine: " << engine_path.string() << std::endl;

    return engine;
}
