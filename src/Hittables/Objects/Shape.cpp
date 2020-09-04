#include <Shape.cuh>

DEV HOST Shape::Shape(const std::shared_ptr<Material> material, const AABB &boundingBox)
    : material{material}, boundingBox{boundingBox} {}

DEV bool Shape::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}