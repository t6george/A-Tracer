#pragma once
#include <Hittable.hpp>

class Texture;

class ConstantVolume : public Hittable
{
    std::shared_ptr<Hittable> boundary;
    std::shared_ptr<Material> phaseFunction;
    const double densityReciprocal;

public:
    ConstantVolume(const std::shared_ptr<Hittable> boundary,
                   const std::shared_ptr<Texture> phaseFunction,
                   const double density);
    ~ConstantVolume() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip = false) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};