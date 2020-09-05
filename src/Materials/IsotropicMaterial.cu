#include <IsotropicMaterial.cuh>
#include <Texture.cuh>
#include <WeightedPdf.cuh>

DEV HOST IsotropicMaterial::IsotropicMaterial(const SharedPointer<Texture> albedo) : Material::Material(albedo) {}

DEV bool IsotropicMaterial::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay = Ray{record.point, Vec3::randomUnitSphereVec(), ray.getTime()};
    record.albedo = albedo->getValue(record.u, record.v, record.point);
    return true;
}
