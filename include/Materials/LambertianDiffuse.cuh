#pragma once

#include <Material.cuh>

class LambertianDiffuse : public Material
{
public:
    DEV HOST LambertianDiffuse(const SharedPointer<Texture> albedo);
    DEV HOST ~LambertianDiffuse() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
