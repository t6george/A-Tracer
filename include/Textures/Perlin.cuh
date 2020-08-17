#pragma once
#include <array>
#include <Vec3.cuh>

class Perlin
{
    static constexpr int pointCount = 256;

    std::array<double, Perlin::pointCount> randomDoubles;
    std::array<Vec3, Perlin::pointCount> randomVectors;
    std::array<int, Perlin::pointCount> permX, permY, permZ;

    DEV void init();

    DEV double trilinearInterpolation(const double c[2][2][2],
                                  double u,
                                  double v,
                                  double w) const;

    DEV double perlinInterpolation(const Vec3 c[2][2][2],
                               double u,
                               double v,
                               double w) const;

public:
    template <typename T, size_t N>
    static DEV void permuteArray(std::array<T, N> &arr);

    DEV HOST Perlin();
    DEV HOST ~Perlin() noexcept = default;

    DEV double getScalarNoise(const Vec3 &point) const;
    DEV double getLaticeVectorNoise(const Vec3 &point) const;

    DEV double getTurbulence(const Vec3 &point, int depth) const;
};