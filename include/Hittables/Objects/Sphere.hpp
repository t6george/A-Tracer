#pragma once
#include <Shape.hpp>
#include <AABB.hpp>

class Sphere : public Shape
{
    const Vec3 center0, center1;
    Vec3 center;
    const double R, time0, time1;

public:
    static void getSphereUV(const Vec3 &p, double &u, double &v);

    Sphere(const Vec3 &center0, const double R,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);

    Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);

    ~Sphere() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record, WeightedPdf &pdf,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    Vec3 blur(const double time) const;
    
    Vec3 genRandomVector(const Vec3& origin) const override;
    double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;
};