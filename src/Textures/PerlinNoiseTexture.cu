#include <PerlinNoiseTexture.cuh>

DEV HOST PerlinNoiseTexture::PerlinNoiseTexture(const double scale, const Vec3 &albedo)
    : NoiseTexture::NoiseTexture{scale, albedo} {}

DEV Vec3 PerlinNoiseTexture::getValue(double u, double v, const Vec3 &point) const
{
    return albedo * .5 * (noise.getLaticeVectorNoise(point * scale) + 1.);
}
