#pragma once

#include <Texture.hpp>

class SolidColor : public Texture
{
    Vec3 color;

public:
    SolidColor() = default;
    SolidColor(const Vec3 &color);

    ~SolidColor() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &p) const override;
};