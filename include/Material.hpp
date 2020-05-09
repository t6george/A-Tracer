#pragma once

#include <Hittable.hpp>

class Material
{
public:
    virtual ~Material() noexcept = default;
    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const = 0;
};