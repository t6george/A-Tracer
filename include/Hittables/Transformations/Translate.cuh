#pragma once
#include <Hittable.cuh>

class Translate : public Hittable
{
    const std::shared_ptr<Hittable> shape;
    Vec3 displacement;

public:
    DEV HOST Translate(const std::shared_ptr<Hittable> shape, const Vec3 &displacement);
    DEV HOST ~Translate() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};