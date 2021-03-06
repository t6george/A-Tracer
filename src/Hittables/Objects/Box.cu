#include <Memory.cuh>

#include <Box.cuh>
#include <AARect.cuh>
#include <FlipFace.cuh>

DEV HOST Box::Box(const Vec3 &p0, const Vec3 &p1,
         const SharedPointer<Material> material)
    : Shape::Shape{material, AABB{p0, p1}},
      minPoint{p0}, maxPoint{p1}
{
    sides.add(mem::MakeShared<AARect<utils::Axis::Z>>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), material));
    sides.add(mem::MakeShared<FlipFace>(
        mem::MakeShared<AARect<utils::Axis::Z>>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), material)));

    sides.add(mem::MakeShared<AARect<utils::Axis::Y>>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), material));
    sides.add(mem::MakeShared<FlipFace>(
        mem::MakeShared<AARect<utils::Axis::Y>>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), material)));

    sides.add(mem::MakeShared<AARect<utils::Axis::X>>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), material));
    sides.add(mem::MakeShared<FlipFace>(
        mem::MakeShared<AARect<utils::Axis::X>>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), material)));
}

DEV Hittable::HitType Box::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    return sides.getCollisionData(ray, record, tMin, tMax, flip);
}