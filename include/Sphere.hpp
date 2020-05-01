#pragma once
#include <Vec3.hpp>

class Ray;

class Sphere
{
    Vec3 center;
    double R;

public:
    Sphere(const Vec3 &center, double R);
    ~Sphere() noexcept = default;
    bool reflectsRay(const Ray &ray) const;
};