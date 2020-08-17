#pragma once
#include <Hittable.cuh>

class Texture;

class ConstantVolume : public Hittable
{
    std::shared_ptr<Hittable> boundary;
    std::shared_ptr<Material> phaseFunction;
    const double densityReciprocal;

public:
    DEV HOST ConstantVolume(const std::shared_ptr<Hittable> boundary,
                   const std::shared_ptr<Texture> phaseFunction,
                   const double density);
    DEV HOST ~ConstantVolume() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};