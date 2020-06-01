#include <LambertianDiffuse.hpp>
#include <Texture.hpp>

LambertianDiffuse::LambertianDiffuse(const std::shared_ptr<Texture> albedo)
    : Material::Material{albedo} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setDirection(record.normal + Vec3::randomUnitSphereVec());
    record.scatteredRay.setTime(ray.getTime());
    record.attenuation = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(record.u, record.v, record.point);

    return true;
}