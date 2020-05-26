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
    NoiseTexture(const double scale, const Vec3 &albedo)
        : scale{scale}, albedo{Vec3::clamp(albedo, 0., 1.)} {}
    virtual ~NoiseTexture() noexcept = default;
};