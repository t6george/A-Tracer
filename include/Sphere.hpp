#pragma once
#include <Shape.hpp>
#include <AABB.hpp>

class Sphere : public Shape
{
    const Vec3 center0, center1;
    Vec3 center;
    double R;
    AABB boundingBox;

public:
    Sphere(const Vec3 &center0, const double R,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);

    Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);

    ~Sphere() noexcept = default;
    bool getCollisionData(const Ray &ray, HitRecord &record,
                          double tMin = -infinity,
                          double tMax = infinity) override;

    const Vec3 &getCenter() const;
    void translate(const double time) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};