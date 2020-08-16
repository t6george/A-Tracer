#pragma once
#include <Material.hpp>

class DiffuseLight : public Material
{
    DEV Vec3 emitCol(const Ray &ray, Hittable::HitRecord &record, 
        const Vec3 &point) const override;

public:
    DEV HOST DiffuseLight(const std::shared_ptr<Texture> emitter);
    DEV HOST ~DiffuseLight() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};