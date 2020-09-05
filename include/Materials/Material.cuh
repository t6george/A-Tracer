#pragma once

#include <Macro.cuh>
#include <Hittable.cuh>

class Texture;
class Vec3;
class WeightedPdf;

class Material
{
protected:
    const SharedPointer<Texture> albedo;

public:
    DEV HOST Material(const SharedPointer<Texture> albedo);
    DEV HOST virtual ~Material() noexcept = default;

    DEV virtual bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const;
    DEV virtual Vec3 emitCol(const Ray &ray, Hittable::HitRecord &record, const Vec3 &point) const;
};
