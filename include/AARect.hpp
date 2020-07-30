#pragma once
#include <Shape.hpp>

template <utils::Axis A>
class AARect : public Shape
{
    double i0, i1, j0, j1, k;

    void solveForTime(const Ray &ray, double &t) const;
    void getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const;
    void setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const;
    AABB computeBoundingBox(const double i0, const double i1, const double j0,
                            const double j1, const double k) const;

public:
    AARect(const double i0, const double i1, const double j0,
           const double j1, const double k,
           const std::shared_ptr<Material> material);
    ~AARect() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip = false) override;
};