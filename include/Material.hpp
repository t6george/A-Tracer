#pragma once

#include <Hittable.hpp>

class Texture;
class Vec3;

class Material
{
protected:
    const std::shared_ptr<Texture> albedo;

public:
    Material(const std::shared_ptr<Texture> albedo);
    virtual ~Material() noexcept = default;

    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const = 0;
    virtual Vec3 emitCol(double u, double v, const Vec3 &point) const;
};