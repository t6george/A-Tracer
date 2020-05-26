#include <NoiseTexture.hpp>
#include <Utils.hpp>

NoiseTexture::NoiseTexture(const double scale, const Vec3 &albedo)
    : scale{scale}, albedo{clamp(albedo, 0., 1.)} {}

Vec3 NoiseTexture::getValue(const double u, const double v, const Vec3 &p) const
{
    // return albedo * noise.getNoise(p * scale);
    // return albedo * noise.getTurbulence(p * scale);
    // return albedo * .5 * (noise.getNoise(p * scale) + 1.);
    return albedo * .5 * (sin(scale * p.z() + 10 * noise.getTurbulence(p)) + 1.);
}