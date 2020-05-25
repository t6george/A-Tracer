#pragma once
#include <limits>
#include <cstdlib>
#include <cmath>
#include <Vec3.hpp>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double deg_to_rad(double degrees)
{
    return degrees * pi / 180.;
}

inline double random_double(double min = 0., double max = 1.)
{
    return min + (max - min) * (rand() / (RAND_MAX + 1.));
}

inline int random_int(int min = 0, int max = 2)
{
    return static_cast<int>(random_double(static_cast<double>(min), static_cast<double>(max)));
}

inline Vec3 random_color(Vec3 min = Vec3{0., 0., 0.}, Vec3 max = Vec3{255., 255., 255.})
{
    return Vec3{random_double(min[0], max[0]), random_double(min[1], max[1]), random_double(min[2], max[2])};
}

inline double clamp(double x, double min, double max)
{
    return fmin(max, fmax(min, x));
}

inline Vec3 random_unit_sphere_vec()
{
    double a = random_double(0, 2 * pi);
    double z = random_double(-1., 1.);
    double r = sqrt(1. - z * z);
    return Vec3{r * cos(a), r * sin(a), z};
}

inline Vec3 random_unit_circle_vec()
{
    Vec3 vec;
    vec[0] = random_double();
    vec[1] = random_double(0., sqrt(1. - vec[0] * vec[0]));
    return vec;
}

inline double schlick(double cos, double relfectiveIndex)
{
    double r0 = (1 - relfectiveIndex) / (1 + relfectiveIndex);
    r0 *= r0;
    return r0 + (1. - r0) * pow(1 - cos, 5);
}