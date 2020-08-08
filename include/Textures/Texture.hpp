#pragma once
#include <Vec3.hpp>

class Texture
{
public:
    Texture() = default;
    virtual ~Texture() noexcept = default;

    virtual Vec3 getValue(double u, double v, const Vec3 &point) const = 0;
};