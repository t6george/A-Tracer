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

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip) override;

    const Vec3 &getCenter() const;
    void blur(const double time);
};