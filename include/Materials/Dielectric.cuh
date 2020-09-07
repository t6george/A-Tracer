#pragma once
#include <Material.cuh>

class Dielectric : public Material
{
    const double reflectiveIndex;

public:
    DEV HOST Dielectric(const double reflectiveIndex);
    DEV HOST ~Dielectric() noexcept = default;
    
    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
