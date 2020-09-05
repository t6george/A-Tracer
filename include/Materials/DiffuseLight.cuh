#pragma once
#include <Material.cuh>

class DiffuseLight : public Material
{
    DEV Vec3 emitCol(const Ray &ray, Hittable::HitRecord &record, 
        const Vec3 &point) const override;

public:
    HOST DiffuseLight(const SharedPointer<Texture> emitter);
    HOST ~DiffuseLight() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
