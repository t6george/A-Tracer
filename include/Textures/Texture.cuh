#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class Texture
{
public:
    HOST Texture() = default;
    HOST virtual ~Texture() noexcept = default;

    DEV virtual Vec3 getValue(double u, double v, const Vec3 &point) const = 0;
};
