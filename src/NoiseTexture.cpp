#include <cmath>
#include <NoiseTexture.hpp>

NoiseTexture::NoiseTexture(const double scale, const Vec3 &albedo)
    : scale{scale}, albedo{Vec3::clamp(albedo, 0., 1.)} {}

Vec3 NoiseTexture::getValue(const double u, const double v, const Vec3 &point) const
{
    // return albedo * noise.getNoise(point * scale);
    // return albedo * noise.getTurbulence(point * scale);
    // return albedo * .5 * (noise.getNoise(point * scale) + 1.);
    return albedo * .5 * (sin(scale * point.z() + 10. * noise.getTurbulence(point)) + 1.);
}