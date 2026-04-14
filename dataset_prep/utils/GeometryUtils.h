#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "dataset_prep/DatasetPrepTypes.h"

namespace dataset_prep {

cv::Point2f Rotate2D(const cv::Point2f& point, float rot_rad);
cv::Matx23f GenTransFromPatchCv(float c_x,
                                float c_y,
                                float src_width,
                                float src_height,
                                int dst_width,
                                int dst_height,
                                float scale,
                                float rot_deg,
                                bool inv = false);

cv::Matx34f BuildProjectionMatrix(const CameraCalibration& calibration);
bool TriangulatePointDLT(const std::vector<cv::Matx34f>& projections,
                         const std::vector<cv::Point2f>& points,
                         cv::Point3f* out_point);
bool ProjectWorldPointToImage(const CameraCalibration& calibration,
                              const cv::Point3f& point_world,
                              cv::Point2f* out_pixel);

cv::Vec3f SmoothAxisAngleRotation(const cv::Vec3f& previous_axis_angle,
                                  const cv::Vec3f& current_axis_angle,
                                  float alpha,
                                  float max_step_radians);
void SmoothAxisAngleBlocks(const std::vector<float>& previous_values,
                           std::vector<float>* current_values,
                           float alpha,
                           float max_step_radians);
cv::Vec3f SmoothTranslationStep(const cv::Vec3f& previous_translation,
                                const cv::Vec3f& current_translation,
                                float alpha,
                                float max_step);

}  // namespace dataset_prep
