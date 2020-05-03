#pragma once

#include <cstddef>
#include <Vec3.hpp>
#include <Ray.hpp>
#include <Utils.hpp>

class Hittable
{
public:
    struct HitRecord
    {
        double t;
        Vec3 point;
        Vec3 normal;
        bool isInFront;

        inline void setLightPosition(const Ray &ray)
        {
            isInFront = ray.direction().o(normal) < 0.;
            if (!isInFront)
            {
                normal = -normal;
            }
        }
    };

    virtual ~Hittable() noexcept = default;
    virtual bool getCollisionData(const Ray &ray, HitRecord &record,
                                  double tMin = -infinity,
                                  double tMax = infinity) const = 0;
};