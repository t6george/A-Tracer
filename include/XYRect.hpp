#pragma once
#include <Shape.hpp>

class XYRect : public Shape
{
    double x0, x1, y0, y1, z;

public:
    XYRect(const double x0, const double x1, const double y0,
           const double y1, const double z,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);
    ~XYRect() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity) override;

    virtual void translate(const double time) override;
};