#pragma once
#include <NoiseTexture.cuh>

class MarbleTexture : public NoiseTexture
{
    const int turbulence;

public:
    HOST MarbleTexture(const double scale = 1.,
                  const Vec3 &albedo = Vec3{1., 1., 1.},
                  const int turbulence = 7);

    HOST ~MarbleTexture() noexcept = default;

    DEV Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
