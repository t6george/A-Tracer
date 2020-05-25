#include <LambertianDiffuse.hpp>
#include <Texture.hpp>

LambertianDiffuse::LambertianDiffuse(const std::shared_ptr<Texture> albedo)
    : Material::Material{albedo} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.resetOrigin(record.point);
    record.scatteredRay.resetDirection(record.normal + random_unit_sphere_vec());
    record.scatteredRay.setTime(ray.time());
    record.attenuation = albedo->getValue(record.u, record.v, record.point);
    return true;
}