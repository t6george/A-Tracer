#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class OrthonormalBasis
{
    Vec3 axes[3];
public:
    HOST OrthonormalBasis(const Vec3& sample);
    HOST ~OrthonormalBasis() noexcept = default;

    DEV Vec3 operator[](int i) const;

    DEV HOST Vec3 getU() const;
    DEV HOST Vec3 getV() const;
    DEV HOST Vec3 getW() const;

    DEV Vec3 getVec(double a, double b, double c) const;
    DEV Vec3 getVec(Vec3 v) const;
};
