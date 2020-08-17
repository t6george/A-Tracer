#pragma once

#include <Material.cuh>

class LambertianDiffuse : public Material
{
public:
    DEV HOST LambertianDiffuse(const std::shared_ptr<Texture> albedo);
    DEV HOST ~LambertianDiffuse() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};