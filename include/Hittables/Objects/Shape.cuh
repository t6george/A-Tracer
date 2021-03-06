#pragma once

#include <Hittable.cuh>
#include <AABB.cuh>

class Shape : public Hittable
{
protected:
    const SharedPointer<Material> material;
    AABB boundingBox;

    DEV HOST Shape(const SharedPointer<Material> material, const AABB &boundingBox);
    DEV HOST virtual ~Shape() noexcept = default;

public:
    DEV virtual bool getBoundingBox(double time0, double time1, AABB &box) const override;
};
