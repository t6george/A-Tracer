#include <NoiseTexture.hpp>
#include <Utils.hpp>

NoiseTexture::NoiseTexture(const Vec3 &albedo) : albedo{clamp(albedo, 0., 1.)} {}

Vec3 NoiseTexture::getValue(const double u, const double v, const Vec3 &p) const
{
    return albedo * noise.getNoise(p);
}