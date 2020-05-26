#pragma once
#include <NoiseTexture.hpp>

class TurbulentTexture : public NoiseTexture
{
    const int turbulence;

public:
    TurbulentTexture(const double scale = 1.,
                     const Vec3 &albedo = Vec3{1., 1., 1.},
                     const int turbulence = 7);

    ~TurbulentTexture() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};
