#pragma once
#include <limits>
#include <cstdlib>
#include <cmath>
#include <Vec3.hpp>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degToRad(double degrees)
{
    return degrees * pi / 180.;
}

inline double random_double()
{
    return rand() / (RAND_MAX + 1.);
}

inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max)
{
    return fmin(max, fmax(min, x));
}

inline Vec3 random_unit_vec()
{
    double a = random_double(0, 2 * pi);
    double z = random_double(-1., 1.);
    double r = sqrt(1. - z * z);
    return Vec3{r * cos(a), r * sin(a), z};
}