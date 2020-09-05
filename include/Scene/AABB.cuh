#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class Ray;

class AABB
{
    Vec3 minPoint, maxPoint;

public:
    DEV HOST static AABB combineAABBs(const AABB &b1, const AABB &b2);
    HOST AABB(const Vec3 &minPoint = Vec3{}, const Vec3 &maxPoint = Vec3{});
    HOST ~AABB() noexcept = default;

    DEV const Vec3 &getMinPoint() const;
    DEV const Vec3 &getMaxPoint() const;

    DEV void setMinPoint(const Vec3 &v);
    DEV void setMaxPoint(const Vec3 &v);

    DEV bool passesThrough(const Ray &ray, double tmin, double tmax) const;
};
