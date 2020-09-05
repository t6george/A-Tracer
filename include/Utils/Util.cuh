#pragma once
#include <limits>
#include <cstdlib>
#include <cmath>

#include <Macro.cuh>

namespace utils
{
    const double infinity = std::numeric_limits<double>::infinity();

    const double pi = 3.1415926535897932385;

    DEV HOST inline double deg_to_rad(double degrees)
    {
        return degrees * pi / 180.;
    }

    DEV HOST inline double random_double(double min = 0., double max = 1.)
    {
#ifdef __CUDACC__
        return 0.;
#else	
        return min + (max - min) * (rand() / (RAND_MAX + 1.));
#endif
    }

    DEV HOST inline int random_int(int min = 0, int max = 2)
    {
        return static_cast<int>(random_double(static_cast<double>(min), static_cast<double>(max)));
    }

    DEV HOST inline double clamp(const double x, const double min, const double max)
    {
        return fmin(max, fmax(min, x));
    }

    DEV HOST inline double schlick(double cos, double relfectiveIndex)
    {
        double r0 = (1 - relfectiveIndex) / (1 + relfectiveIndex);
        r0 *= r0;
        return r0 + (1. - r0) * pow(1 - cos, 5);
    }

    template <typename T1, typename T2>
    DEV HOST void swap(T1& o1, T2& o2)
    {
        T1 tmp = o1;
	o1 = o2;
	o2 = tmp;
    }

    enum class Axis
    {
        X,
        Y,
        Z
    };
} // namespace utils
