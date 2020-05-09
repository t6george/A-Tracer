#pragma once
#include <Material.hpp>

class Metal : public Material
{
    const Vec3 albedo;

public:
    Metal(Vec3 color);
    virtual ~Metal() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};