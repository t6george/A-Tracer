#include <Metal.hpp>
#include <Utils.hpp>
#include <Texture.hpp>

Metal::Metal(const std::shared_ptr<Texture> albedo, const double fuzz)
    : Material::Material{albedo}, fuzz{utils::clamp(fuzz, 0., 1.)} {}

bool Metal::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.resetOrigin(record.point);
    record.scatteredRay.resetDirection(ray.direction().getUnitVector().reflect(record.normal) +
                                       Vec3::randomUnitSphereVec() * fuzz);
    record.scatteredRay.setTime(ray.time());
    record.attenuation = albedo->getValue(record.u, record.v, record.point);
    return record.scatteredRay.direction().o(record.normal) > 0.;
}