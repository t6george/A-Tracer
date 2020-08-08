#pragma once
#include <Material.hpp>

class DiffuseLight : public Material
{
    Vec3 emitCol(const Ray &ray, Hittable::HitRecord &record, 
        const Vec3 &point) const override;

public:
    DiffuseLight(const std::shared_ptr<Texture> emitter);
    ~DiffuseLight() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};