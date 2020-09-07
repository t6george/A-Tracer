#pragma once
#include <NoiseTexture.cuh>

class SmoothNoiseTexture : public NoiseTexture
{
public:
    DEV HOST SmoothNoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    DEV HOST ~SmoothNoiseTexture() noexcept = default;

    DEV Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
