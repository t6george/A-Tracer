#pragma once
#include <Material.hpp>

class DiffuseLight : public Material
{
    Vec3 emitCol(double u, double v, const Vec3 &point) const override;

public:
    DiffuseLight(const std::shared_ptr<Texture> emitter);
    ~DiffuseLight() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};