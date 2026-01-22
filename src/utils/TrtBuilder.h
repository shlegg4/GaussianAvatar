#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

void CheckCuda(cudaError_t err, const char* msg);
int64_t Volume(const nvinfer1::Dims& dims);

TrtUniquePtr<nvinfer1::ICudaEngine> BuildEngineFromOnnx(const std::string& model_path,
                                                        TrtLogger& logger);
