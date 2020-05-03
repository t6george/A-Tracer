#pragma once
#include <Hittable.hpp>

class Sphere : public Hittable
{
    Vec3 center;
    double R;

public:
    Sphere(const Vec3 &center, double R);
    ~Sphere() noexcept = default;

    bool getCollisionData(const Ray &ray, HitRecord &record,
                          double tMin = -infinity,
                          double tMax = infinity) const override;

    const Vec3 &getCenter() const;
};