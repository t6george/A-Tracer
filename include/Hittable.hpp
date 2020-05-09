#pragma once

#include <cstddef>
#include <memory>

#include <Vec3.hpp>
#include <Ray.hpp>
#include <Utils.hpp>

class Material;

class Hittable
{
protected:
    const Material &material;

    Hittable(const Material &material);

public:
    struct HitRecord
    {
        double t;
        Vec3 point;
        Vec3 normal;
        bool isInFront;
        Ray reflectedRay;

        void setLightPosition(const Ray &ray);
    };

    virtual ~Hittable() noexcept = default;
    virtual bool getCollisionData(const Ray &ray, HitRecord &record,
                                  double tMin = -infinity,
                                  double tMax = infinity) const = 0;
};