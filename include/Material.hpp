#pragma once

#include <Hittable.hpp>
#include <Vec3.hpp>

#include <iostream>

class Material
{
protected:
    const Vec3 albedo;

public:
    Material(const Vec3 &color) : albedo{color / 255.} {}
    virtual ~Material() noexcept = default;
    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const = 0;
};