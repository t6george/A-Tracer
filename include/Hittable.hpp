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
    const std::shared_ptr<Material> material;
    const double time0, time1;

    Hittable(const std::shared_ptr<Material> material,
             const double t0, const double t1);

public:
    struct HitRecord
    {
        double t;
        Vec3 point;
        Vec3 normal;
        Vec3 attenuation;
        bool isInFront;
        Ray scatteredRay;

        void setLightPosition(const Ray &ray);
    };

    virtual ~Hittable() noexcept = default;
    virtual bool getCollisionData(const Ray &ray, HitRecord &record,
                                  double tMin = -infinity,
                                  double tMax = infinity) = 0;
    virtual void translate(const double time) = 0;
};