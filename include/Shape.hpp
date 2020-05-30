#pragma once
#include <Hittable.hpp>
#include <AABB.hpp>

class Shape : public Hittable
{
protected:
    const std::shared_ptr<Material> material;
    const double time0, time1;
    AABB boundingBox;

    Shape(const std::shared_ptr<Material> material, const double t0,
          const double t1, const AABB &boundingBox);
    virtual ~Shape() noexcept = default;

public:
    virtual void translate(const double time) = 0;
    virtual bool getBoundingBox(double time0, double time1, AABB &box) const override;
};