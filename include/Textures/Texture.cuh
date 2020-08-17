#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class Texture
{
public:
    DEV HOST Texture() = default;
    virtual DEV HOST ~Texture() noexcept = default;

    virtual DEV Vec3 getValue(double u, double v, const Vec3 &point) const = 0;
};