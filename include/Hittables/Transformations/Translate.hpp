#pragma once
#include <Hittable.hpp>

class Translate : public Hittable
{
    const std::shared_ptr<Hittable> shape;
    Vec3 displacement;

public:
    Translate(const std::shared_ptr<Hittable> shape, const Vec3 &displacement);
    ~Translate() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};