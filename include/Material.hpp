#pragma once

#include <Hittable.hpp>

class Material
{
public:
    virtual void scatterRay(const Ray &ray, Hittable::HitRecord &record) const = 0;
};