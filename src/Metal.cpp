#include <Metal.hpp>
#include <Utils.hpp>

Metal::Metal(Vec3 color, double fuzz) : Material::Material{color}, fuzz{clamp(fuzz, 0., 1.)} {}

bool Metal::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.scatteredRay.resetOrigin(record.point);
    record.scatteredRay.resetDirection(ray.direction().getUnitVector().reflect(record.normal) +
                                       random_unit_sphere_vec() * fuzz);
    record.scatteredRay.setTime(ray.time());
    record.attenuation = albedo;
    return record.scatteredRay.direction().o(record.normal) > 0.;
}