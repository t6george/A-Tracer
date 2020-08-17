#pragma once

#include <Texture.cuh>

class SolidColor : public Texture
{
    const Vec3 color;

public:
    DEV HOST SolidColor() = default;
    DEV HOST SolidColor(const Vec3 &color);
    DEV HOST SolidColor(const double r, const double g, const double b);

    DEV HOST ~SolidColor() noexcept = default;

    DEV Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};