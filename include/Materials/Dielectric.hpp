#pragma once
#include <Material.hpp>

class Dielectric : public Material
{
    const double reflectiveIndex;

public:
    DEV HOST Dielectric(const double reflectiveIndex);
    DEV HOST ~Dielectric() noexcept = default;
    
    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};