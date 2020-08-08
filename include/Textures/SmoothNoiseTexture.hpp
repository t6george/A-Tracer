#pragma once
#include <NoiseTexture.hpp>

class SmoothNoiseTexture : public NoiseTexture
{
public:
    SmoothNoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    ~SmoothNoiseTexture() noexcept = default;

    Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
