#include <LambertianDiffuse.hpp>
#include <Texture.hpp>
#include <OrthonormalBasis.hpp>
#include <CosinePdf.hpp>
#include <HittablePdf.hpp>
#include <WeightedPdf.hpp>
#include <AARect.hpp>

LambertianDiffuse::LambertianDiffuse(const std::shared_ptr<Texture> albedo)
    : Material::Material{albedo} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.isSpecular = false;
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setTime(ray.getTime());

    record.albedo = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(ray, record, record.point);

    return true;
}