#pragma once

#include <Material.cuh>

class LambertianDiffuse : public Material
{
public:
    HOST LambertianDiffuse(const SharedPointer<Texture> albedo);
    HOST ~LambertianDiffuse() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
