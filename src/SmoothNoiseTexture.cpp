#include <SmoothNoiseTexture.hpp>

SmoothNoiseTexture::SmoothNoiseTexture(const double scale, const Vec3 &albedo)
    : NoiseTexture::NoiseTexture{scale, albedo} {}

Vec3 SmoothNoiseTexture::getValue(const double u, const double v, const Vec3 &point) const
{
    return albedo * noise.getScalarNoise(point * scale);
}