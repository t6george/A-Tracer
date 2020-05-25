#include <SolidColor.hpp>

SolidColor::SolidColor(const Vec3 &color) : color{color} {}

Vec3 SolidColor::getValue(const double u, const double v, const Vec3 &p) const { return color; }