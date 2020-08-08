#pragma once
#include <Material.hpp>

class Dielectric : public Material
{
    const double reflectiveIndex;

public:
    Dielectric(const double reflectiveIndex);

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};