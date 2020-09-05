#include <LambertianDiffuse.cuh>
#include <Texture.cuh>
#include <OrthonormalBasis.cuh>
#include <CosinePdf.cuh>
#include <HittablePdf.cuh>
#include <WeightedPdf.cuh>
#include <AARect.cuh>

DEV HOST LambertianDiffuse::LambertianDiffuse(const SharedPointer<Texture> albedo)
    : Material::Material{albedo} {}

DEV bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.isSpecular = false;
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setTime(ray.getTime());

    record.albedo = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(ray, record, record.point);

    return true;
}