#pragma once
#include <vector>
#include <memory>

#include <Hittable.hpp>

class Ray;

class HittableList : public Hittable
{
    std::vector<std::shared_ptr<Hittable>> hittables;

public:
    HittableList() = default;
    ~HittableList() noexcept = default;

    bool getCollisionData(const Ray &ray, Hittable::HitRecord &record,
                          double tMin = -infinity,
                          double tMax = infinity) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;

    void add(std::shared_ptr<Hittable> hittable);
    void clear();
};