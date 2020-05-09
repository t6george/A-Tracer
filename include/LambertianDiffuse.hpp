#pragma once

#include <Material.hpp>
#include <Vec3.hpp>

class LambertianDiffuse : public Material
{
    const Vec3 albedo;

public:
    LambertianDiffuse(Vec3 color);
    ~LambertianDiffuse() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};