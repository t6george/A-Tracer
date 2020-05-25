#pragma once

#include <Hittable.hpp>

class Texture;

class Material
{
protected:
    const std::shared_ptr<Texture> albedo;

public:
    Material(const std::shared_ptr<Texture> albedo) : albedo{albedo} {}
    virtual ~Material() noexcept = default;
    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const = 0;
};