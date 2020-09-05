#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class Texture
{
public:
    DEV HOST Texture() = default;
    DEV HOST virtual ~Texture() noexcept = default;

    DEV virtual Vec3 getValue(double u, double v, const Vec3 &point) const = 0;
};
