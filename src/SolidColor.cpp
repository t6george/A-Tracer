#include <SolidColor.hpp>
#include <Utils.hpp>

SolidColor::SolidColor(const Vec3 &color) : color{clamp(color, 0., 1.)} {}

SolidColor::SolidColor(const double r, const double g, const double b) : color{Vec3{r, g, b}} {}

Vec3 SolidColor::getValue(const double u, const double v, const Vec3 &p) const { return color; }