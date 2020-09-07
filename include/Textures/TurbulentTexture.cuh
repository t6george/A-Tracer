#pragma once
#include <NoiseTexture.cuh>

class TurbulentTexture : public NoiseTexture
{
    const int turbulence;

public:
    DEV HOST TurbulentTexture(const double scale = 1.,
                     const Vec3 &albedo = Vec3{1., 1., 1.},
                     const int turbulence = 7);

    DEV HOST ~TurbulentTexture() noexcept = default;

    DEV Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};
