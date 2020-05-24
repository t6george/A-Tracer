#pragma once
#include <Hittable.hpp>

class Shape : public Hittable
{
protected:
    const std::shared_ptr<Material> material;
    const double time0, time1;

    Shape(const std::shared_ptr<Material> material,
          const double t0, const double t1);
    virtual ~Shape() noexcept = default;

public:
    virtual void translate(const double time) = 0;
};