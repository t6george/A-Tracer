#pragma once
#include <Material.cuh>

class Dielectric : public Material
{
    const double reflectiveIndex;

public:
    HOST Dielectric(const double reflectiveIndex);
    HOST ~Dielectric() noexcept = default;
    
    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
