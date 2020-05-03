#pragma once
#include <Hittable.hpp>
#include <vector>
#include <memory>

class HittableList : public Hittable
{
    std::vector<std::shared_ptr<Hittable>> hittables;

public:
    HittableList() = default;
    ~HittableList() noexcept = default;

    bool getCollisionData(const Ray &ray, HitRecord &record,
                          double tMin = -infinity,
                          double tMax = infinity) const override;
    void add(std::shared_ptr<Hittable> hittable);
    void clear();
};