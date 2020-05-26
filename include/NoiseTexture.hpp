#pragma once
#include <Texture.hpp>
#include <Perlin.hpp>

class NoiseTexture : public Texture
{
    const double scale;
    const Vec3 albedo;
    const Perlin noise;

public:
    NoiseTexture(const double scale = 1., const Vec3 &albedo = Vec3{1., 1., 1.});
    ~NoiseTexture() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &p) const override;
};