#pragma once

#include <opencv2/core.hpp>

namespace dataset_prep {

bool SanitizeBBox(const cv::Rect2f& bbox, int img_width, int img_height, cv::Rect2f* out_bbox);
bool ProcessBBox(const cv::Rect2f& bbox,
                 int img_width,
                 int img_height,
                 const cv::Size& input_shape,
                 float ratio,
                 cv::Rect2f* out_bbox);

bool GeneratePatchImage(const cv::Mat& image,
                        const cv::Rect2f& bbox,
                        float scale,
                        float rot_deg,
                        bool do_flip,
                        const cv::Size& out_shape,
                        cv::Mat* out_patch,
                        cv::Matx23f* out_trans,
                        cv::Matx23f* out_inv_trans);

cv::Mat ApplyCropMatte(const cv::Mat& crop_image, const cv::Mat& crop_matte);
cv::Mat CropSquareWithPaddingAndResize(const cv::Mat& src, int x, int y, int size, int target_res);
cv::Point2f ApplyAffinePoint(const cv::Matx23f& affine, const cv::Point2f& point);

}  // namespace dataset_prep
