#include <cmath>

#include <CheckerTexture.hpp>

CheckerTexture::CheckerTexture(const std::shared_ptr<Texture> tex1,
                               const std::shared_ptr<Texture> tex2, const Vec3 &scale)
    : tex1{tex1}, tex2{tex2}, scale{scale} {}

CheckerTexture::CheckerTexture(const std::shared_ptr<Texture> tex1,
                               const std::shared_ptr<Texture> tex2,
                               const double x, const double y, const double z)
    : CheckerTexture::CheckerTexture{tex1, tex2, Vec3{x, y, z}} {}

Vec3 CheckerTexture::getValue(const double u, const double v, const Vec3 &p) const
{
    double sines = sin(scale.x() * p.x()) * sin(scale.y() * p.y()) * sin(scale.z() * p.z());
    return sines >= 0. ? tex1->getValue(u, v, p) : tex2->getValue(u, v, p);
}