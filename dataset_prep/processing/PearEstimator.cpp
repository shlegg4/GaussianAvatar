#include "dataset_prep/processing/PearEstimator.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/OnnxRuntimeCudaProvider.h"

namespace dataset_prep {
namespace {

constexpr int kPearInputSize = 256;

std::string ToLower(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

int FindNameIndex(const std::vector<std::string>& names, const std::string& pattern) {
    const std::string pattern_lower = ToLower(pattern);
    for (size_t index = 0; index < names.size(); ++index) {
        if (ToLower(names[index]).find(pattern_lower) != std::string::npos) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

std::vector<float> PreprocessPearImage(const cv::Mat& image) {
    cv::Mat resized;
    if (image.cols == kPearInputSize && image.rows == kPearInputSize) {
        resized = image.clone();
    } else {
        cv::resize(image,
                   resized,
                   cv::Size(kPearInputSize, kPearInputSize),
                   0.0,
                   0.0,
                   cv::INTER_LINEAR);
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(3 * kPearInputSize * kPearInputSize));
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < kPearInputSize; ++row) {
            const float* row_ptr = channels[channel].ptr<float>(row);
            for (int col = 0; col < kPearInputSize; ++col) {
                input_data.push_back(row_ptr[col]);
            }
        }
    }
    return input_data;
}

size_t GetTensorElementCount(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        return 0u;
    }

    const auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t element_count = 1u;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0u;
        }
        element_count *= static_cast<size_t>(dim);
    }
    return element_count;
}

bool CopyTensorToVector(const Ort::Value& tensor, std::vector<float>* out_values) {
    if (out_values == nullptr) {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    if (element_count == 0u) {
        return false;
    }

    const float* data = tensor.GetTensorData<float>();
    if (data == nullptr) {
        return false;
    }

    out_values->assign(data, data + element_count);
    return true;
}

bool ConvertPoseTensorToAxisAngle(const Ort::Value& tensor, std::vector<float>* out_axis_angles) {
    if (out_axis_angles == nullptr) {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    if (element_count == 0u) {
        return false;
    }

    const float* pose_data = tensor.GetTensorData<float>();
    if (pose_data == nullptr) {
        return false;
    }

    if ((element_count % 9u) == 0u) {
        const size_t num_matrices = element_count / 9u;
        out_axis_angles->assign(num_matrices * 3u, 0.0f);

        for (size_t matrix_index = 0; matrix_index < num_matrices; ++matrix_index) {
            const float* src = pose_data + matrix_index * 9u;
            const cv::Matx33f rot_mat(
                src[0], src[1], src[2],
                src[3], src[4], src[5],
                src[6], src[7], src[8]);
            cv::Vec3f axis_angle;
            cv::Rodrigues(rot_mat, axis_angle);

            (*out_axis_angles)[matrix_index * 3u + 0u] = axis_angle[0];
            (*out_axis_angles)[matrix_index * 3u + 1u] = axis_angle[1];
            (*out_axis_angles)[matrix_index * 3u + 2u] = axis_angle[2];
        }
        return true;
    }

    if ((element_count % 3u) == 0u) {
        out_axis_angles->assign(pose_data, pose_data + element_count);
        return true;
    }

    return false;
}

bool ExtractCameraTranslation(const Ort::Value& tensor, std::vector<float>* out_translation) {
    if (out_translation == nullptr) {
        return false;
    }

    const size_t element_count = GetTensorElementCount(tensor);
    if (element_count == 0u) {
        return false;
    }

    const float* data = tensor.GetTensorData<float>();
    if (data == nullptr) {
        return false;
    }

    if (element_count >= 12u) {
        *out_translation = {data[3], data[7], data[11]};
        return true;
    }
    if (element_count >= 3u) {
        *out_translation = {data[0], data[1], data[2]};
        return true;
    }
    return false;
}

void AssignVectorPrefix(const std::vector<float>& source, size_t count, std::vector<float>* target) {
    if (target == nullptr) {
        return;
    }
    const size_t take_count = std::min(count, source.size());
    target->assign(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(take_count));
}

}  // namespace

class PearEstimator::Impl {
public:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info{nullptr};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    int image_input_index = -1;
    int camera_output_index = -1;
    int global_pose_output_index = -1;
    int body_pose_output_index = -1;
    int left_hand_pose_output_index = -1;
    int right_hand_pose_output_index = -1;
    int jaw_pose_output_index = -1;
    int betas_output_index = -1;
    int expression_output_index = -1;
};

PearEstimator::PearEstimator(const Options& options)
    : options_(options) {}

PearEstimator::~PearEstimator() = default;

bool PearEstimator::Initialize() {
    impl_ = std::make_unique<Impl>();

    try {
        impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dataset_prep_pear");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (options_.use_cuda) {
            onnxruntime_utils::TryAppendCudaExecutionProvider(session_options);
        }

        const std::wstring model_path(options_.model_path.begin(), options_.model_path.end());
        impl_->session = std::make_unique<Ort::Session>(*impl_->env, model_path.c_str(), session_options);
        impl_->memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = impl_->session->GetInputCount();
        impl_->input_names.clear();
        for (size_t index = 0; index < input_count; ++index) {
            auto name = impl_->session->GetInputNameAllocated(index, allocator);
            impl_->input_names.emplace_back(name.get());
        }

        const size_t output_count = impl_->session->GetOutputCount();
        impl_->output_names.clear();
        for (size_t index = 0; index < output_count; ++index) {
            auto name = impl_->session->GetOutputNameAllocated(index, allocator);
            impl_->output_names.emplace_back(name.get());
        }

        impl_->image_input_index = FindNameIndex(impl_->input_names, "image");
        if (impl_->image_input_index < 0 && !impl_->input_names.empty()) {
            impl_->image_input_index = 0;
        }

        impl_->camera_output_index = FindNameIndex(impl_->output_names, "camera_rt");
        if (impl_->camera_output_index < 0) {
            impl_->camera_output_index = FindNameIndex(impl_->output_names, "camera");
        }
        impl_->global_pose_output_index = FindNameIndex(impl_->output_names, "global_pose");
        if (impl_->global_pose_output_index < 0) {
            impl_->global_pose_output_index = FindNameIndex(impl_->output_names, "global");
        }
        impl_->body_pose_output_index = FindNameIndex(impl_->output_names, "body_pose");
        impl_->left_hand_pose_output_index = FindNameIndex(impl_->output_names, "left_hand_pose");
        impl_->right_hand_pose_output_index = FindNameIndex(impl_->output_names, "right_hand_pose");
        impl_->jaw_pose_output_index = FindNameIndex(impl_->output_names, "flame_jaw_pose");
        if (impl_->jaw_pose_output_index < 0) {
            impl_->jaw_pose_output_index = FindNameIndex(impl_->output_names, "jaw");
        }
        impl_->betas_output_index = FindNameIndex(impl_->output_names, "body_shape");
        if (impl_->betas_output_index < 0) {
            impl_->betas_output_index = FindNameIndex(impl_->output_names, "betas");
        }
        impl_->expression_output_index = FindNameIndex(impl_->output_names, "body_expression");
        if (impl_->expression_output_index < 0) {
            impl_->expression_output_index = FindNameIndex(impl_->output_names, "expression");
        }
    } catch (const Ort::Exception&) {
        impl_.reset();
        return false;
    }

    return IsReady();
}

bool PearEstimator::IsReady() const {
    return impl_ != nullptr &&
           impl_->session != nullptr &&
           impl_->image_input_index >= 0 &&
           impl_->camera_output_index >= 0 &&
           impl_->global_pose_output_index >= 0 &&
           impl_->body_pose_output_index >= 0 &&
           impl_->left_hand_pose_output_index >= 0 &&
           impl_->right_hand_pose_output_index >= 0 &&
           impl_->jaw_pose_output_index >= 0 &&
           impl_->betas_output_index >= 0 &&
           impl_->expression_output_index >= 0;
}

bool PearEstimator::Estimate(const cv::Mat& crop_image, SmplxResult* out_result) const {
    if (!IsReady() || out_result == nullptr || crop_image.empty()) {
        return false;
    }

    std::vector<float> image_input = PreprocessPearImage(crop_image);
    std::vector<int64_t> image_shape = {1, 3, kPearInputSize, kPearInputSize};

    const char* image_input_name = impl_->input_names[static_cast<size_t>(impl_->image_input_index)].c_str();
    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        impl_->memory_info,
        image_input.data(),
        image_input.size(),
        image_shape.data(),
        image_shape.size());

    std::vector<const char*> output_names;
    output_names.reserve(impl_->output_names.size());
    for (const auto& name : impl_->output_names) {
        output_names.push_back(name.c_str());
    }

    std::vector<Ort::Value> outputs;
    try {
        outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                      &image_input_name,
                                      &image_tensor,
                                      1u,
                                      output_names.data(),
                                      output_names.size());
    } catch (const Ort::Exception&) {
        return false;
    }

    const auto output_count = static_cast<int>(outputs.size());
    if (impl_->camera_output_index >= output_count ||
        impl_->global_pose_output_index >= output_count ||
        impl_->body_pose_output_index >= output_count ||
        impl_->left_hand_pose_output_index >= output_count ||
        impl_->right_hand_pose_output_index >= output_count ||
        impl_->jaw_pose_output_index >= output_count ||
        impl_->betas_output_index >= output_count ||
        impl_->expression_output_index >= output_count) {
        return false;
    }

    SmplxResult result;
    std::vector<float> betas;
    std::vector<float> expression;
    if (!ExtractCameraTranslation(outputs[static_cast<size_t>(impl_->camera_output_index)],
                                  &result.camera_translation) ||
        !ConvertPoseTensorToAxisAngle(outputs[static_cast<size_t>(impl_->global_pose_output_index)],
                                      &result.global_orient) ||
        !ConvertPoseTensorToAxisAngle(outputs[static_cast<size_t>(impl_->body_pose_output_index)],
                                      &result.body_pose) ||
        !ConvertPoseTensorToAxisAngle(outputs[static_cast<size_t>(impl_->left_hand_pose_output_index)],
                                      &result.left_hand_pose) ||
        !ConvertPoseTensorToAxisAngle(outputs[static_cast<size_t>(impl_->right_hand_pose_output_index)],
                                      &result.right_hand_pose) ||
        !ConvertPoseTensorToAxisAngle(outputs[static_cast<size_t>(impl_->jaw_pose_output_index)],
                                      &result.jaw_pose) ||
        !CopyTensorToVector(outputs[static_cast<size_t>(impl_->betas_output_index)], &betas) ||
        !CopyTensorToVector(outputs[static_cast<size_t>(impl_->expression_output_index)], &expression)) {
        return false;
    }

    AssignVectorPrefix(betas, 10u, &result.betas);
    AssignVectorPrefix(expression, 10u, &result.expression);
    *out_result = std::move(result);
    return true;
}

}  // namespace dataset_prep
