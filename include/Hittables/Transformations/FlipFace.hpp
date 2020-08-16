#pragma once
#include <Hittable.hpp>

class AABB;

class FlipFace : public Hittable
{
    const std::shared_ptr<Hittable> hittable;

public:
    DEV HOST FlipFace(const std::shared_ptr<Hittable> hittable);
    DEV HOST ~FlipFace() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};