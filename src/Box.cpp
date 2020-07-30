#include <memory>

#include <Box.hpp>
#include <AARect.hpp>
#include <FlipFace.hpp>

Box::Box(const Vec3 &p0, const Vec3 &p1,
         const std::shared_ptr<Material> material)
    : Shape::Shape{material, AABB{p0, p1}},
      minPoint{p0}, maxPoint{p1}
{
    sides.add(std::make_shared<AARect<utils::Axis::Z>>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), material));
    sides.add(std::make_shared<AARect<utils::Axis::Z>>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), material));

    sides.add(std::make_shared<AARect<utils::Axis::Y>>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), material));
    sides.add(std::make_shared<AARect<utils::Axis::Y>>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), material));

    sides.add(std::make_shared<AARect<utils::Axis::X>>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), material));
    sides.add(std::make_shared<AARect<utils::Axis::X>>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), material));
}

Hittable::HitType Box::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax, bool flip)
{
    return sides.getCollisionData(ray, record, tMin, tMax, flip);
}