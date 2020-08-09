#pragma once
#include <Material.hpp>

class Metal : public Material
{
    double fuzz;

public:
    Metal(const std::shared_ptr<Texture> albedo, const double fuzz = 0.);
    virtual ~Metal() noexcept = default;

    bool scatterRay(const Ray &ray, Hittable::HitRecord &record,
        WeightedPdf& pdf) const override;
};