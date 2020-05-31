#pragma once
#include <Shape.hpp>
// #include <AARect.hpp>
// #include <Material.hpp>
// #include <AABB.hpp>
enum class Axis
{
    X,
    Y,
    Z
};

template <Axis A>
class AARect : public Shape
{
    double i0, i1, j0, j1, k;

    void solveForTime(const Ray &ray, double &t) const;
    void getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const;
    void setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const;

public:
    AARect(const double i0, const double i1, const double j0,
           const double j1, const double k,
           const std::shared_ptr<Material> material,
           const double t0 = 0., const double t1 = 1.);
    ~AARect() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity) override;

    virtual void translate(const double time) override;
};