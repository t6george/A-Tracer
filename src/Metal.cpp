#include <Metal.hpp>
#include <Utils.hpp>

Metal::Metal(Vec3 color, double fuzz) : albedo{color / 255.}, fuzz{clamp(fuzz, 0., 1.)} {}

bool Metal::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.reflectedRay.resetOrigin(record.point);
    record.reflectedRay.resetDirection(ray.direction().getUnitVector().reflect(record.normal) +
                                       random_unit_vec() * fuzz);
    record.attenuation = albedo;
    return record.reflectedRay.direction().o(record.normal) > 0.;
}