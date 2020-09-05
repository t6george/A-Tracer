#pragma once
#include <Hittable.cuh>
#include <AABB.cuh>

class Shape : public Hittable
{
protected:
    const std::shared_ptr<Material> material;
    AABB boundingBox;

    DEV HOST Shape(const std::shared_ptr<Material> material, const AABB &boundingBox);
    DEV HOST virtual ~Shape() noexcept = default;

public:
    DEV virtual bool getBoundingBox(double time0, double time1, AABB &box) const override;
};
