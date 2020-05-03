#pragma once
#include <cstddef>
#include <Vec3.hpp>

class Ray;

class Surface
{
public:
    struct HitRecord
    {
        double t;
        Vec3 point;
        Vec3 normal;
    };

    virtual bool getCollisionData(const Ray &ray, HitRecord &record,
                                  double tMin = -std::numeric_limits<double>::max(),
                                  double tMax = std::numeric_limits<double>::max()) const = 0;
};