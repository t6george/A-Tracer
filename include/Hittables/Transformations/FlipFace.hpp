#pragma once
#include <Hittable.hpp>

class AABB;

class FlipFace : public Hittable
{
    const std::shared_ptr<Hittable> hittable;

public:
    FlipFace(const std::shared_ptr<Hittable> hittable);
    ~FlipFace() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record, WeightedPdf &pdf,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};