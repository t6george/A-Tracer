#pragma once
#include <Shape.cuh>
#include <AABB.cuh>

class Sphere : public Shape
{
    const Vec3 center0, center1;
    Vec3 center;
    const double R, time0, time1;

public:
    static DEV void getSphereUV(const Vec3 &p, double &u, double &v);

    HOST Sphere(const Vec3 &center0, const double R,
           const SharedPointer<Material> material,
           const double t0 = 0., const double t1 = 1.);

    HOST Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
           const SharedPointer<Material> material,
           const double t0 = 0., const double t1 = 1.);

    HOST ~Sphere() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV Vec3 blur(const double time) const;
    
    DEV Vec3 genRandomVector(const Vec3& origin) const override;
    DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;
};
