#pragma once

#include <Macro.cuh>
#include <Hittable.cuh>

class Texture;
class Vec3;
class WeightedPdf;

class Material
{
protected:
    const std::shared_ptr<Texture> albedo;

public:
    Material(const std::shared_ptr<Texture> albedo);
    virtual ~Material() noexcept = default;

    virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const;
    virtual Vec3 emitCol(const Ray &ray, Hittable::HitRecord &record, const Vec3 &point) const;
};