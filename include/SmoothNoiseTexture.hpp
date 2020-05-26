#pragma once
#include <NoiseTexture.hpp>

class SmoothNoiseTexture : public NoiseTexture
{
public:
    SmoothNoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    ~SmoothNoiseTexture() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};
