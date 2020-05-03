#include <Sphere.hpp>
#include <Ray.hpp>
#include <math.h>

Sphere::Sphere(const Vec3 &center, double R) : center{center}, R{R} {}

// double Sphere::pointOfIncidence(const Ray &ray) const
// {
//     Vec3 los = ray.origin() - center;

//     double a = ray.direction().o(ray.direction());
//     double half_b = ray.direction().o(los);
//     double c = los.o(los) - R * R;
//     double disc = half_b * half_b - a * c;

//     if (disc < 0.0)
//     {
//         return -1.;
//     }
//     else
//     {
//         return (-half_b - sqrt(disc)) / a;
//     }
// }

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
            return true;
        }
        t = (-half_b + disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            record.setLightPosition(ray);
            return true;
        }
    }
    return false;
}

const Vec3 &Sphere::getCenter() const
{
    return center;
}