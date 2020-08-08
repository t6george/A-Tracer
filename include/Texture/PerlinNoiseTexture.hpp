#pragma once
#include <NoiseTexture.hpp>

class PerlinNoiseTexture : public NoiseTexture
{
public:
    PerlinNoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    ~PerlinNoiseTexture() noexcept = default;

    Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
