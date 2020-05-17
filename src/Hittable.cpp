#include <Hittable.hpp>

void Hittable::HitRecord::setLightPosition(const Ray &ray)
{
    isInFront = ray.direction().o(normal) < 0.;
    if (!isInFront)
    {
        normal = -normal;
    }
}

Hittable::Hittable(const std::shared_ptr<Material> material) : material{material} {}