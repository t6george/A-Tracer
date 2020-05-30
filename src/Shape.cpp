#include <Shape.hpp>

Shape::Shape(const std::shared_ptr<Material> material, const double t0,
             const double t1, const AABB &boundingBox)
    : material{material}, time0{t0}, time1{t1}, boundingBox{boundingBox} {}

bool Shape::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}