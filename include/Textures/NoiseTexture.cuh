#pragma once
#include <Texture.cuh>
#include <Perlin.cuh>

class NoiseTexture : public Texture
{
protected:
    const double scale;
    const Vec3 albedo;
    const Perlin noise;

public:
    HOST NoiseTexture(const double scale, const Vec3 &albedo)
        : scale{scale}, albedo{Vec3::clamp(albedo, 0., 1.)} {}
    HOST virtual ~NoiseTexture() noexcept = default;
};
