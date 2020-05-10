#include <Dielectric.hpp>

Dielectric::Dielectric(const double reflectiveIndex) : reflectiveIndex{reflectiveIndex} {}

bool Dielectric::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    double n_over_nprime = reflectiveIndex;
    if (record.isInFront)
    {
        n_over_nprime = 1. / n_over_nprime;
    }

    double cos = fmin(1., ray.direction().getUnitVector().o(record.normal));
    double sin = sqrt(1.0 - cos * cos);

    if (n_over_nprime * sin > 1. || random_double() < schlick(cos, n_over_nprime))
    {
        record.reflectedRay.resetDirection(ray.direction().getUnitVector().reflect(
            record.normal));
    }
    else
    {
        record.reflectedRay.resetDirection(ray.direction().getUnitVector().refract(
            record.normal, n_over_nprime));
    }

    record.attenuation = Vec3{1., 1., 1.};
    record.reflectedRay.resetOrigin(record.point);

    return true;
}