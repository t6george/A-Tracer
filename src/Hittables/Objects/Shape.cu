#include <Shape.cuh>

DEV HOST Shape::Shape(const SharedPointer<Material> material, const AABB &boundingBox)
    : material{material}, boundingBox{boundingBox} {}

DEV bool Shape::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}
