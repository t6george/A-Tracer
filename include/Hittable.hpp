#pragma once

#include <cstddef>
#include <memory>

#include <Vec3.hpp>
#include <Ray.hpp>
#include <Utils.hpp>

class Material;
class AABB;

class Hittable
{
protected:
    Hittable() = default;

    virtual ~Hittable() noexcept = default;

public:
    struct HitRecord
    {
        double t, u, v;
        Vec3 point;
        Vec3 normal;
        Vec3 attenuation;
        Vec3 emitted;
        bool isInFront;
        Ray scatteredRay;

        inline void setLightPosition(const Ray &ray)
        {
            isInFront = ray.getDirection().o(normal) < 0.;
            if (!isInFront)
            {
                normal = -normal;
            }
        }
    };

    enum class HitType
    {
        NO_HIT,
        HIT_NO_SCATTER,
        HIT_SCATTER
    };

    virtual HitType getCollisionData(const Ray &ray, HitRecord &record,
                                     double tMin = -utils::infinity,
                                     double tMax = utils::infinity) = 0;

    virtual bool getBoundingBox(double time0, double time1, AABB &box) const = 0;
};