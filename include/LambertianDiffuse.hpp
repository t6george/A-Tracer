#pragma once

#include <Material.hpp>
#include <Vec3.hpp>

class LambertianDiffuse : public Material
{
    const Vec3 albedo;

public:
    LambertianDiffuse(Vec3 albedo);
    ~LambertianDiffuse() noexcept = default;

    void scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};