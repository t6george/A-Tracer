#pragma once
#include <array>
#include <Vec3.hpp>

class Perlin
{
    static constexpr int pointCount = 256;

    // std::array<double, Perlin::pointCount> randomDoubles;
    std::array<Vec3, Perlin::pointCount> randomVectors;
    std::array<int, Perlin::pointCount> permX, permY, permZ;

    void init();

    double trilinearInterpolation(const double c[2][2][2],
                                  double u,
                                  double v,
                                  double w) const;

    double perlinInterpolation(const Vec3 c[2][2][2],
                               double u,
                               double v,
                               double w) const;

public:
    template <typename T, size_t N>
    static void permuteArray(std::array<T, N> &arr);

    Perlin();
    ~Perlin() noexcept = default;

    double getNoise(const Vec3 &point) const;
    double getTurbulence(const Vec3 &point, int depth = 7) const;
};