#pragma once

#include <Vec3.cuh>

class Perlin
{
    static constexpr int pointCount = 256;

    double randomDoubles[Perlin::pointCount];
    Vec3 randomVectors[Perlin::pointCount];

    int permX[Perlin::pointCount];
    int permY[Perlin::pointCount];
    int permZ[Perlin::pointCount];

    DEV HOST void init();

    DEV double trilinearInterpolation(const double c[2][2][2],
                                  double u,
                                  double v,
                                  double w) const;

    DEV double perlinInterpolation(const Vec3 c[2][2][2],
                               double u,
                               double v,
                               double w) const;

public:
    DEV HOST static void permuteArray(int* arr, int N);

    DEV HOST Perlin();
    DEV HOST ~Perlin() noexcept = default;

    DEV double getScalarNoise(const Vec3 &point) const;
    DEV double getLaticeVectorNoise(const Vec3 &point) const;

    DEV double getTurbulence(const Vec3 &point, int depth) const;
};
