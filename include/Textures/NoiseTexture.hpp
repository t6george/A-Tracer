#pragma once
#include <Texture.hpp>
#include <Perlin.hpp>

class NoiseTexture : public Texture
{
protected:
    const double scale;
    const Vec3 albedo;
    const Perlin noise;

public:
    DEV HOST NoiseTexture(const double scale, const Vec3 &albedo)
        : scale{scale}, albedo{Vec3::clamp(albedo, 0., 1.)} {}
    virtual DEV HOST ~NoiseTexture() noexcept = default;
};