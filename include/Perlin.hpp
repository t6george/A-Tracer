#pragma once
#include <array>

class Vec3;

class Perlin
{
    static constexpr int pointCount = 256;

    std::array<double, Perlin::pointCount> randomDoubles;
    std::array<int, Perlin::pointCount> permX, permY, permZ;

    void init();

public:
    template <typename T, size_t N>
    static void permuteArray(std::array<T, N> &arr);

    Perlin();
    ~Perlin() noexcept = default;

    double getNoise(const Vec3 &point) const;
};