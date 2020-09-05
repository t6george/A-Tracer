#include <SolidColor.cuh>

HOST SolidColor::SolidColor(const Vec3 &color) : color{color} {}

HOST SolidColor::SolidColor(const double r, const double g, const double b) : color{Vec3{r, g, b}} {}

DEV Vec3 SolidColor::getValue(const double u, const double v, const Vec3 &point) const { return color; }
