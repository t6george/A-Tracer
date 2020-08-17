#pragma once
#include <array>

#include <Macro.cuh>
#include <Vec3.cuh>

class OrthonormalBasis
{
    std::array<Vec3, 3> axes;

public:
    OrthonormalBasis(const Vec3& sample);
    ~OrthonormalBasis() noexcept = default;

    Vec3 operator[](int i) const;

    Vec3 getU() const;
    Vec3 getV() const;
    Vec3 getW() const;

    Vec3 getVec(double a, double b, double c) const;
    Vec3 getVec(Vec3 v) const;
};