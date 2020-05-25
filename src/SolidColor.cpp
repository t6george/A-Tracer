#include <SolidColor.hpp>
#include <Vec3.hpp>

SolidColor::SolidColor(const Vec3 &color) : color{color} {}

SolidColor::SolidColor(const double r, const double g, const double b) : color{Vec3{r, g, b}} {}

Vec3 SolidColor::getValue(const double u, const double v, const Vec3 &p) const { return color; }