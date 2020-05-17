#include <math.h>

#include <Sphere.hpp>
#include <Ray.hpp>
#include <Material.hpp>

Sphere::Sphere(const Vec3 &center, double R, const std::shared_ptr<Material> material)
    : Hittable::Hittable{material}, center{center}, R{R} {}

bool Sphere::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax) const
{
    Vec3 los = ray.origin() - center;

    double a = ray.direction().o(ray.direction());
    double half_b = ray.direction().o(los);
    double c = los.o(los) - R * R;
    double disc = half_b * half_b - a * c;
    double disc_root, t;

    if (disc > 0.)
    {
        disc_root = sqrt(disc);
        t = (-half_b - disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            record.setLightPosition(ray);
            return material->scatterRay(ray, record);
        }
        t = (-half_b + disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            record.setLightPosition(ray);
            return material->scatterRay(ray, record);
        }
    }
    return false;
}

const Vec3 &Sphere::getCenter() const
{
    return center;
}