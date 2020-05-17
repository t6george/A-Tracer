#pragma once
#include <Material.hpp>

class Metal : public Material
{
    double fuzz;

public:
    Metal(Vec3 color, double fuzz = 0.);
    virtual ~Metal() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};