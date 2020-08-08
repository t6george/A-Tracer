#include <TurbulentTexture.hpp>

TurbulentTexture::TurbulentTexture(const double scale, const Vec3 &albedo, const int turbulence)
    : NoiseTexture::NoiseTexture{scale, albedo}, turbulence{turbulence} {}

Vec3 TurbulentTexture::getValue(double u, double v, const Vec3 &point) const
{
    return albedo * noise.getTurbulence(point * scale, turbulence);
}