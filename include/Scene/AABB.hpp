#pragma once

#include <Macro.hpp>
#include <Vec3.hpp>

class Ray;

class AABB
{
    Vec3 minPoint, maxPoint;

public:
    static DEV HOST AABB combineAABBs(const AABB &b1, const AABB &b2);
    DEV HOST AABB(const Vec3 &minPoint = Vec3{}, const Vec3 &maxPoint = Vec3{});
    DEV HOST ~AABB() noexcept = default;

    DEV const Vec3 &getMinPoint() const;
    DEV const Vec3 &getMaxPoint() const;

    DEV void setMinPoint(const Vec3 &v);
    DEV void setMaxPoint(const Vec3 &v);

    DEV bool passesThrough(const Ray &ray, double tmin, double tmax) const;
};