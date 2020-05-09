#include <LambertianDiffuse.hpp>

LambertianDiffuse::LambertianDiffuse(Vec3 albedo) : albedo{albedo} {}

void LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
}