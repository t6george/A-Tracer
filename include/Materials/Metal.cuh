#pragma once
#include <Material.cuh>

class Metal : public Material
{
    double fuzz;

public:
    DEV HOST Metal(const SharedPointer<Texture> albedo, const double fuzz = 0.);
    DEV HOST virtual ~Metal() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
