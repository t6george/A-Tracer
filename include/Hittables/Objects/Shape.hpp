#pragma once
#include <Hittable.hpp>
#include <AABB.hpp>

class Shape : public Hittable
{
protected:
    const std::shared_ptr<Material> material;
    AABB boundingBox;

    Shape(const std::shared_ptr<Material> material, const AABB &boundingBox);
    virtual ~Shape() noexcept = default;

public:
    virtual bool getBoundingBox(double time0, double time1, AABB &box) const override;
};