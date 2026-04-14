#include "dataset_prep/utils/ImageUtils.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "dataset_prep/utils/GeometryUtils.h"

namespace dataset_prep {

bool SanitizeBBox(const cv::Rect2f& bbox, int img_width, int img_height, cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    const float x1 = std::max(0.0f, bbox.x);
    const float y1 = std::max(0.0f, bbox.y);
    const float x2 = std::min(static_cast<float>(img_width - 1),
                              x1 + std::max(0.0f, bbox.width - 1.0f));
    const float y2 = std::min(static_cast<float>(img_height - 1),
                              y1 + std::max(0.0f, bbox.height - 1.0f));
    if (bbox.width * bbox.height <= 0.0f || x2 <= x1 || y2 <= y1) {
        return false;
    }

    *out_bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    return true;
}

bool ProcessBBox(const cv::Rect2f& bbox,
                 int img_width,
                 int img_height,
                 const cv::Size& input_shape,
                 float ratio,
                 cv::Rect2f* out_bbox) {
    if (out_bbox == nullptr) {
        return false;
    }

    cv::Rect2f clipped_bbox;
    if (!SanitizeBBox(bbox, img_width, img_height, &clipped_bbox)) {
        return false;
    }

    float w = clipped_bbox.width;
    float h = clipped_bbox.height;
    const float c_x = clipped_bbox.x + w * 0.5f;
    const float c_y = clipped_bbox.y + h * 0.5f;
    const float aspect_ratio = static_cast<float>(input_shape.width) /
                               static_cast<float>(input_shape.height);
    if (w > aspect_ratio * h) {
        h = w / aspect_ratio;
    } else if (w < aspect_ratio * h) {
        w = h * aspect_ratio;
    }

    const float expanded_w = w * ratio;
    const float expanded_h = h * ratio;
    *out_bbox = cv::Rect2f(c_x - expanded_w * 0.5f,
                           c_y - expanded_h * 0.5f,
                           expanded_w,
                           expanded_h);
    return true;
}

bool GeneratePatchImage(const cv::Mat& image,
                        const cv::Rect2f& bbox,
                        float scale,
                        float rot_deg,
                        bool do_flip,
                        const cv::Size& out_shape,
                        cv::Mat* out_patch,
                        cv::Matx23f* out_trans,
                        cv::Matx23f* out_inv_trans) {
    if (out_patch == nullptr || image.empty()) {
        return false;
    }

    cv::Mat src = image;
    float bb_c_x = bbox.x + bbox.width * 0.5f;
    const float bb_c_y = bbox.y + bbox.height * 0.5f;
    const float bb_width = bbox.width;
    const float bb_height = bbox.height;

    if (do_flip) {
        cv::flip(image, src, 1);
        bb_c_x = static_cast<float>(image.cols) - bb_c_x - 1.0f;
    }

    const cv::Matx23f trans = GenTransFromPatchCv(
        bb_c_x, bb_c_y, bb_width, bb_height,
        out_shape.width, out_shape.height, scale, rot_deg, false);
    cv::warpAffine(src,
                   *out_patch,
                   cv::Mat(trans),
                   out_shape,
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    if (out_trans != nullptr) {
        *out_trans = trans;
    }
    if (out_inv_trans != nullptr) {
        *out_inv_trans = GenTransFromPatchCv(
            bb_c_x, bb_c_y, bb_width, bb_height,
            out_shape.width, out_shape.height, scale, rot_deg, true);
    }
    return true;
}

cv::Mat ApplyCropMatte(const cv::Mat& crop_image, const cv::Mat& crop_matte) {
    if (crop_image.empty() || crop_matte.empty()) {
        return crop_image.clone();
    }

    cv::Mat matte_resized;
    if (crop_matte.size() != crop_image.size()) {
        cv::resize(crop_matte, matte_resized, crop_image.size(), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        matte_resized = crop_matte;
    }

    cv::Mat matte_f32;
    if (matte_resized.type() == CV_32F) {
        matte_f32 = matte_resized;
    } else {
        matte_resized.convertTo(matte_f32, CV_32F, 1.0 / 255.0);
    }

    cv::Mat matte_bgr;
    if (matte_f32.channels() == 1) {
        cv::cvtColor(matte_f32, matte_bgr, cv::COLOR_GRAY2BGR);
    } else {
        matte_bgr = matte_f32;
    }

    cv::Mat crop_f32;
    crop_image.convertTo(crop_f32, CV_32F, 1.0 / 255.0);

    cv::Mat matted_f32 = crop_f32.mul(matte_bgr);
    cv::Mat matted_crop;
    matted_f32.convertTo(matted_crop, crop_image.type(), 255.0);
    return matted_crop;
}

cv::Mat CropSquareWithPaddingAndResize(const cv::Mat& src, int x, int y, int size, int target_res) {
    if (src.empty() || size <= 0 || target_res <= 0) {
        return {};
    }

    const cv::Rect roi(x, y, size, size);
    const cv::Rect image_bounds(0, 0, src.cols, src.rows);
    const cv::Rect valid_roi = roi & image_bounds;

    cv::Mat square = cv::Mat::zeros(size, size, src.type());
    if (valid_roi.area() > 0) {
        const cv::Rect dst_roi(valid_roi.x - roi.x, valid_roi.y - roi.y, valid_roi.width, valid_roi.height);
        src(valid_roi).copyTo(square(dst_roi));
    }

    cv::Mat resized;
    cv::resize(square,
               resized,
               cv::Size(target_res, target_res),
               0.0,
               0.0,
               size > target_res ? cv::INTER_AREA : cv::INTER_CUBIC);
    return resized;
}

cv::Point2f ApplyAffinePoint(const cv::Matx23f& affine, const cv::Point2f& point) {
    return cv::Point2f(affine(0, 0) * point.x + affine(0, 1) * point.y + affine(0, 2),
                       affine(1, 0) * point.x + affine(1, 1) * point.y + affine(1, 2));
}

}  // namespace dataset_prep
