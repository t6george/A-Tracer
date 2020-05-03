#pragma once
#include <Surface.hpp>

class Sphere : public Surface
{
    Vec3 center;
    double R;

public:
    Sphere(const Vec3 &center, double R);
    ~Sphere() noexcept = default;

    bool getCollisionData(const Ray &ray, HitRecord &record,
                          double tMin = -std::numeric_limits<double>::max(),
                          double tMax = std::numeric_limits<double>::max()) const override;

    const Vec3 &getCenter() const;
};