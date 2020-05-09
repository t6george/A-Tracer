#include <LambertianDiffuse.hpp>

LambertianDiffuse::LambertianDiffuse(Vec3 color) : albedo{color / 255.999} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.reflectedRay.resetOrigin(record.point);
    record.reflectedRay.resetDirection(record.normal + random_unit_vec());
    record.attenuation = albedo;
    return true;
}