#include <Sphere.hpp>
#include <Ray.hpp>

Sphere::Sphere(const Vec3 &center, double R) : center{center}, R{R} {}

bool Sphere::reflectsRay(const Ray &ray) const
{
    Vec3 los = ray.origin() - center;

    double a = ray.direction().o(ray.direction());
    double b = ray.direction().o(los) * 2.;
    double c = los.o(los) - R * R;
    return b * b - 4 * a * c >= 0.;
}