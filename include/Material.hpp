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

    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const { return false; }
    virtual void scatterPdf(const Ray &ray, Hittable::HitRecord &record) const { }
    virtual Vec3 emitCol(double u, double v, const Vec3 &point) const;
};