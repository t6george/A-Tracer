#pragma once
#include <Shape.cuh>
#include <HittableList.cuh>

class Box : public Shape
{
    Vec3 minPoint, maxPoint;
    HittableList sides;

public:
    HOST Box(const Vec3 &p0, const Vec3 &p1,
        const SharedPointer<Material> material);

    HOST ~Box() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;
};