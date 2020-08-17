#pragma once
#include <Shape.cuh>
#include <HittableList.cuh>

class Box : public Shape
{
    Vec3 minPoint, maxPoint;
    HittableList sides;

public:
    DEV HOST Box(const Vec3 &p0, const Vec3 &p1,
        const std::shared_ptr<Material> material);

    DEV HOST ~Box() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;
};