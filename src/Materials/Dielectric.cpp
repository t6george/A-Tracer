#include <Dielectric.cuh>
#include <SolidColor.cuh>
#include <WeightedPdf.cuh>

DEV HOST Dielectric::Dielectric(const double reflectiveIndex)
    : Material::Material{std::make_shared<SolidColor>(Vec3{1., 1., 1.})},
      reflectiveIndex{reflectiveIndex} {}

DEV bool Dielectric::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    double n_over_nprime = reflectiveIndex;
    if (record.isInFront)
    {
        n_over_nprime = 1. / n_over_nprime;
    }

    double cos = fmin(1., (-ray.getDirection().getUnitVector()).o(record.normal));
    double sin = sqrt(1. - cos * cos);

    if (n_over_nprime * sin > 1. || utils::random_double() < utils::schlick(cos, n_over_nprime))
    {
        record.scatteredRay.setDirection(ray.getDirection().getUnitVector().reflect(
            record.normal));
    }
    else
    {
        record.scatteredRay.setDirection(ray.getDirection().getUnitVector().refract(
            record.normal, n_over_nprime));
    }

    record.albedo = albedo->getValue(record.u, record.v, Vec3{});
    record.scatteredRay.setOrigin(record.point);
    record.scatteredRay.setTime(ray.getTime());
    record.emitted = emitCol(ray, record, Vec3{});

    record.isSpecular = true;

    return true;
}