#pragma once
#include <Material.cuh>

class Metal : public Material
{
    double fuzz;

public:
    HOST Metal(const SharedPointer<Texture> albedo, const double fuzz = 0.);
    HOST virtual ~Metal() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};
