#include <LambertianDiffuse.hpp>

LambertianDiffuse::LambertianDiffuse(Vec3 color) : albedo{color / 255.} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.resetOrigin(record.point);
    record.scatteredRay.resetDirection(record.normal + random_unit_vec());
    record.attenuation = albedo;
    return true;
}