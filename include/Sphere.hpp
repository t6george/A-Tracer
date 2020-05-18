#pragma once
#include <Hittable.hpp>

class Sphere : public Hittable
{
    const Vec3 center0, center1;
    Vec3 center;
    double R;

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
};