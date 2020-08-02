#include <LambertianDiffuse.hpp>
#include <Texture.hpp>
#include <OrthonormalBasis.hpp>
#include <CosinePdf.hpp>
#include <HittablePdf.hpp>
#include <WeightedPdf.hpp>
#include <WeightedPdf.hpp>
#include <AARect.hpp>

LambertianDiffuse::LambertianDiffuse(const std::shared_ptr<Texture> albedo)
    : Material::Material{albedo} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setTime(ray.getTime());

    record.albedo = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(ray, record, record.point);

    // WeightedPdf pdf{std::make_shared<CosinePdf>(record.normal), 
    //     std::make_shared<HittablePdf>(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., 
    //     std::make_shared<Material>(nullptr)), record.point), .5};

    // record.scatteredRay.setDirection(pdf.genRandomVector());

    // record.samplePdf = pdf.eval(record.scatteredRay.getDirection());

    // record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);

    return true;
}

void LambertianDiffuse::scatterPdf(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);
}