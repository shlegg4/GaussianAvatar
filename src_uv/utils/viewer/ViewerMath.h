#pragma once

#include <tuple>

#include <torch/torch.h>

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4
{
    float m[16] = {};
};

struct Quat
{
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 Normalize(const Vec3 &v);
Vec3 LoadCoeff(const float *sh, int idx);
Vec3 AddScaled(const Vec3 &base, const Vec3 &coeff, float scale);
Vec3 ComputeColorFromSH(int degree, const float *sh, const Vec3 &pos, const Vec3 &cam);

Mat4 Identity();
Mat4 Multiply(const Mat4 &a, const Mat4 &b);
Mat4 Perspective(float fovy_deg, float aspect, float z_near, float z_far);
Mat4 LookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up);
Mat4 RotationYawPitch(float yaw, float pitch);

torch::Tensor Mat4ToTensorRowMajor(const Mat4 &m, torch::Device device);

Vec3 RotatePoint(const Vec3 &p, const Vec3 &center, float yaw, float pitch);
Vec3 AddVec3(const Vec3 &a, const Vec3 &b);
Vec3 SubVec3(const Vec3 &a, const Vec3 &b);
Vec3 ScaleVec3(const Vec3 &v, float s);
Vec3 CrossVec3(const Vec3 &a, const Vec3 &b);
Vec3 CameraPosFromView(const Mat4 &view);

Quat NormalizeQuat(const Quat &q);
Quat MulQuat(const Quat &a, const Quat &b);
Quat ConjugateQuat(const Quat &q);
Quat AxisAngleQuat(const Vec3 &axis, float angle);
Quat YawPitchQuat(float yaw, float pitch);

Mat4 Mat4FromQuatTranslation(const Quat &q, const Vec3 &t);
Vec3 RotatePointQuat(const Vec3 &p, const Vec3 &center, const Quat &q);
Vec3 RotateVec3Quat(const Vec3 &v, const Quat &q);

torch::Tensor QuatToMat3(const Quat &q, torch::Device device);
torch::Tensor MulQuatTensor(const torch::Tensor &a, const torch::Tensor &b);

std::tuple<torch::Tensor, torch::Tensor, float, float> BuildProjection(float fovy_deg, int width, int height,
                                                                       torch::Device device);
Vec3 TransformPoint(const Mat4 &m, const Vec3 &p);
