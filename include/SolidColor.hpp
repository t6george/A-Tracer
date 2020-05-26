#pragma once

#include <Texture.hpp>

class SolidColor : public Texture
{
    const Vec3 color;

public:
    SolidColor() = default;
    SolidColor(const Vec3 &color);
    SolidColor(const double r, const double g, const double b);

    ~SolidColor() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};