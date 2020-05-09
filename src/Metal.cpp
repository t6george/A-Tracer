#include <Metal.hpp>

Metal::Metal(Vec3 color) : albedo{color / 255.999} {}

bool Metal::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.reflectedRay.resetOrigin(record.point);
    record.reflectedRay.resetDirection(ray.direction().reflect(record.normal).getUnitVector());
    record.attenuation = albedo;
    return record.reflectedRay.direction().o(record.normal) > 0.;
}