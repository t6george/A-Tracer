#pragma once

#include <Texture.cuh>

class SolidColor : public Texture
{
    const Vec3 color;

public:
    HOST SolidColor() = default;
    HOST SolidColor(const Vec3 &color);
    HOST SolidColor(const double r, const double g, const double b);

    HOST ~SolidColor() noexcept = default;

    DEV Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};
