#pragma once

#include <onnxruntime_cxx_api.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace onnxruntime_utils {

inline bool TryAppendCudaExecutionProvider(Ort::SessionOptions& session_options,
                                           int device_id = 0) {
#if defined(_WIN32)
    using AppendCudaFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);

    static const AppendCudaFn append_cuda = []() -> AppendCudaFn {
        const wchar_t* module_names[] = {
            L"onnxruntime.dll",
            L"onnxruntime_providers_shared.dll",
        };
        for (const wchar_t* module_name : module_names) {
            HMODULE module = ::GetModuleHandleW(module_name);
            if (module == nullptr) {
                module = ::LoadLibraryW(module_name);
            }
            if (module == nullptr) {
                continue;
            }

            FARPROC proc =
                ::GetProcAddress(module, "OrtSessionOptionsAppendExecutionProvider_CUDA");
            if (proc != nullptr) {
                return reinterpret_cast<AppendCudaFn>(proc);
            }
        }
        return nullptr;
    }();

    if (append_cuda == nullptr) {
        return false;
    }

    try {
        Ort::ThrowOnError(append_cuda(session_options, device_id));
        return true;
    } catch (const Ort::Exception&) {
        return false;
    }
#else
    (void)session_options;
    (void)device_id;
    return false;
#endif
}

}  // namespace onnxruntime_utils
