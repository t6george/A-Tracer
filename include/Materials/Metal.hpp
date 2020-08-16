#pragma once
#include <Material.hpp>

class Metal : public Material
{
    double fuzz;

public:
    DEV HOST Metal(const std::shared_ptr<Texture> albedo, const double fuzz = 0.);
    DEV HOST virtual ~Metal() noexcept = default;

    DEV bool scatterRay(const Ray &ray, Hittable::HitRecord &record) const override;
};