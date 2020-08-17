#pragma once

#include <cstddef>
#include <memory>

#include <Macro.cuh>
#include <Vec3.cuh>
#include <Ray.cuh>
#include <Util.cuh>

class Material;
class AABB;
class WeightedPdf;

class Hittable
{
protected:
    DEV HOST Hittable() = default;
    virtual DEV HOST ~Hittable() noexcept = default;

public:
    struct HitRecord
    {
        double t, u, v;
        Vec3 point;
        Vec3 normal;
        Vec3 albedo;
        Vec3 emitted;
        bool isInFront;
        Ray scatteredRay;
        double samplePdf;
        double scatterPdf;
        bool isSpecular;

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

    virtual DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const = 0;

    virtual DEV bool getBoundingBox(double time0, double time1, AABB &box) const = 0;

    virtual DEV Vec3 genRandomVector(const Vec3& origin) const { return Vec3{1., 0., 0.}; }
    
    virtual DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const { return 0.; }
};