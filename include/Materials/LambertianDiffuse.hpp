#pragma once

#include <Material.hpp>

class LambertianDiffuse : public Material
{
public:
    LambertianDiffuse(const std::shared_ptr<Texture> albedo);
    ~LambertianDiffuse() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};