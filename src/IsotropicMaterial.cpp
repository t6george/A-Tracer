#include <IsotropicMaterial.hpp>
#include <Texture.hpp>

bool IsotropicMaterial::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay = Ray{record.point, Vec3::randomUnitSphereVec(), ray.getTime()};
    record.attenuation = albedo->getValue(record.u, record.v, record.point);
    return true;
}
