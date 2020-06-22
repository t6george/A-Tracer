#pragma once
#include <Material.hpp>

class IsotropicMaterial : public Material
{
public:
    IsotropicMaterial(const std::shared_ptr<Texture> albedo);
    virtual ~IsotropicMaterial() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};