#pragma once
#include <Material.cuh>

class IsotropicMaterial : public Material
{
public:
    DEV HOST IsotropicMaterial(const std::shared_ptr<Texture> albedo);
    DEV HOST virtual ~IsotropicMaterial() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;};