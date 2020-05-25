#pragma once
#include <Vec3.hpp>

class Texture
{
public:
    Texture() = default;
    virtual ~Texture() noexcept = default;

    virtual Vec3 getValue(const double u, const double v, const Vec3 &p) const = 0;
};