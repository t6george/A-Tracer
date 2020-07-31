#include <LambertianDiffuse.hpp>
#include <Texture.hpp>
#include <OrthonormalBasis.hpp>
#include <CosinePdf.hpp>

LambertianDiffuse::LambertianDiffuse(const std::shared_ptr<Texture> albedo)
    : Material::Material{albedo} {}

bool LambertianDiffuse::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    // OrthonormalBasis onb(record.normal);

    // record.scatteredRay.setOrigin(record.point);
    // //record.scatteredRay.setDirection((record.normal + Vec3::randomUnitSphereVec()).getUnitVector());
    // //record.scatteredRay.setDirection(Vec3::randomUnitHemisphereVec(record.normal));
    // record.scatteredRay.setDirection(onb.getVec(Vec3::randomCosineVec()).getUnitVector());
    // record.scatteredRay.setTime(ray.getTime());
    // record.albedo = albedo->getValue(record.u, record.v, record.point);
    // record.emitted = emitCol(ray, record, record.point);
    // //record.samplePdf = record.normal.o(record.scatteredRay.getDirection()) / utils::pi;
    // //record.samplePdf = .5 / utils::pi;
    // record.samplePdf = onb.getW().o(record.scatteredRay.getDirection()) / utils::pi;
    // scatterPdf(ray, record);

    CosinePdf pdf(record.normal);
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setDirection(pdf.genRandomVector());
    record.scatteredRay.setTime(ray.getTime());
    record.albedo = albedo->getValue(record.u, record.v, record.point);
    record.emitted = emitCol(ray, record, record.point);

    record.samplePdf = pdf.eval(record.scatteredRay.getDirection());
    scatterPdf(ray, record);
    return true;
}

void LambertianDiffuse::scatterPdf(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);
}