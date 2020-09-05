#include <Metal.cuh>
#include <Util.cuh>
#include <Texture.cuh>
#include <WeightedPdf.cuh>

DEV HOST Metal::Metal(const SharedPointer<Texture> albedo, const double fuzz)
    : Material::Material{albedo}, fuzz{utils::clamp(fuzz, 0., 1.)} {}

DEV bool Metal::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.isSpecular = true;
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setDirection(ray.getDirection().getUnitVector().reflect(record.normal) +
                                     Vec3::randomUnitSphereVec() * fuzz);
    record.scatteredRay.setTime(ray.getTime());
    record.albedo = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(ray, record, record.point);

    return record.scatteredRay.getDirection().o(record.normal) > 0.;
}