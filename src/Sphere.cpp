#include <Sphere.hpp>
#include <Ray.hpp>
#include <math.h>

Sphere::Sphere(const Vec3 &center, double R) : center{center}, R{R} {}

double Sphere::pointOfIncidence(const Ray &ray) const
{
    Vec3 los = ray.origin() - center;

    double a = ray.direction().o(ray.direction());
    double half_b = ray.direction().o(los);
    double c = los.o(los) - R * R;
    double disc = half_b * half_b - a * c;

    if (disc < 0.0)
    {
        return -1.;
    }
    else
    {
        return (-half_b - sqrt(disc)) / a;
    }
}

bool Sphere::reflectsRay(const Ray &ray) const
{
    return pointOfIncidence(ray) > 0.;
}

const Vec3 &Sphere::getCenter() const
{
    return center;
}