#pragma once
#include <Shape.cuh>

template <utils::Axis A>
class AARect : public Shape
{
    double i0, i1, j0, j1, k, area;

    DEV void solveForTime(const Ray &ray, double &t) const;
    DEV void getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const;
    DEV void setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const;
    DEV AABB computeBoundingBox(const double i0, const double i1, const double j0,
                            const double j1, const double k) const;

public:
    HOST AARect(const double i0, const double i1, const double j0,
           const double j1, const double k,
           const SharedPointer<Material> material);
    HOST ~AARect() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV Vec3 genRandomVector(const Vec3& origin) const override;
    DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;
};