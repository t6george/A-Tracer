#include <IsotropicMaterial.cuh>
#include <Texture.cuh>
#include <WeightedPdf.cuh>

IsotropicMaterial::IsotropicMaterial(const std::shared_ptr<Texture> albedo) : Material::Material(albedo) {}

bool IsotropicMaterial::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay = Ray{record.point, Vec3::randomUnitSphereVec(), ray.getTime()};
    record.albedo = albedo->getValue(record.u, record.v, record.point);
    return true;
}
