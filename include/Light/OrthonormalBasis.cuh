#pragma once
#include <array>

#include <Macro.cuh>
#include <Vec3.cuh>

class OrthonormalBasis
{
    std::array<Vec3, 3> axes;

public:
    DEV HOST OrthonormalBasis(const Vec3& sample);
    DEV HOST ~OrthonormalBasis() noexcept = default;

    DEV Vec3 operator[](int i) const;

    DEV Vec3 getU() const;
    DEV Vec3 getV() const;
    DEV Vec3 getW() const;

    DEV Vec3 getVec(double a, double b, double c) const;
    DEV Vec3 getVec(Vec3 v) const;
};