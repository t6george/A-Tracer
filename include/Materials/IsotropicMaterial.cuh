#pragma once
#include <Material.cuh>

class IsotropicMaterial : public Material
{
public:
    HOST IsotropicMaterial(const SharedPointer<Texture> albedo);
    HOST virtual ~IsotropicMaterial() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;};
