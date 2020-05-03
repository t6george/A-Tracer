#pragma once
#include <limits>
#include <cstdlib>
#include <cmath>

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