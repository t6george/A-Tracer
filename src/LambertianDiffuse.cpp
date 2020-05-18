#include <LambertianDiffuse.hpp>

LambertianDiffuse::LambertianDiffuse(Vec3 color) : Material::Material{color} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.resetOrigin(record.point);
    record.scatteredRay.resetDirection(record.normal + random_unit_sphere_vec());
    record.scatteredRay.setTime(ray.time());
    record.attenuation = albedo;
    return true;
}