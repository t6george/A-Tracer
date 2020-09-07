#include <SmoothNoiseTexture.cuh>

DEV HOST SmoothNoiseTexture::SmoothNoiseTexture(const double scale, const Vec3 &albedo)
    : NoiseTexture::NoiseTexture{scale, albedo} {}

DEV Vec3 SmoothNoiseTexture::getValue(double u, double v, const Vec3 &point) const
{
    return albedo * noise.getScalarNoise(point * scale);
}
