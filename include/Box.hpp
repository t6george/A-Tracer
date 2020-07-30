#pragma once
#include <Shape.hpp>
#include <HittableList.hpp>

class Box : public Shape
{
    Vec3 minPoint, maxPoint;
    HittableList sides;

public:
    Box(const Vec3 &p0, const Vec3 &p1,
        const std::shared_ptr<Material> material);

    ~Box() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip = false) override;
};