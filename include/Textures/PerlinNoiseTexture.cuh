#pragma once
#include <NoiseTexture.cuh>

class PerlinNoiseTexture : public NoiseTexture
{
public:
    HOST PerlinNoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    HOST ~PerlinNoiseTexture() noexcept = default;

    DEV Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
