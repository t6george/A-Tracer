#include <cmath>

#include <MarbleTexture.cuh>

DEV HOST MarbleTexture::MarbleTexture(const double scale, const Vec3 &albedo, const int turbulence)
    : NoiseTexture::NoiseTexture{scale, albedo}, turbulence{turbulence} {}

DEV Vec3 MarbleTexture::getValue(double u, double v, const Vec3 &point) const
{
    return albedo * .5 * (sin(scale * point.z() + 10. * noise.getTurbulence(point, turbulence)) + 1.);
}
