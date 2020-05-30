#pragma once
#include <Material.hpp>

class DiffuseLight : public Material
{
public:
    DiffuseLight(const std::shared_ptr<Texture> emitter);
    ~DiffuseLight() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
    Vec3 emitCol(double u, double v, const Vec3 &point) const override;
};