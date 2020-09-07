#include <cmath>

#include <CheckerTexture.cuh>

DEV HOST CheckerTexture::CheckerTexture(const SharedPointer<Texture> tex1,
                               const SharedPointer<Texture> tex2, const Vec3 &scale)
    : tex1{tex1}, tex2{tex2}, scale{scale} {}

DEV HOST CheckerTexture::CheckerTexture(const SharedPointer<Texture> tex1,
                               const SharedPointer<Texture> tex2,
                               const double x, const double y, const double z)
    : CheckerTexture::CheckerTexture{tex1, tex2, Vec3{x, y, z}} {}

DEV Vec3 CheckerTexture::getValue(double u, double v, const Vec3 &point) const
{
    double sines = sin(scale.x() * point.x()) * sin(scale.y() * point.y()) * sin(scale.z() * point.z());
    return sines >= 0. ? tex1->getValue(u, v, point) : tex2->getValue(u, v, point);
}
