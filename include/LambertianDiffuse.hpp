#pragma once

#include <Material.hpp>
#include <Vec3.hpp>

class LambertianDiffuse : public Material
{
public:
    LambertianDiffuse(Vec3 color);
    ~LambertianDiffuse() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};