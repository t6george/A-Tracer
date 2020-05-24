#pragma once
#include <Vec3.hpp>

class Ray;

class AABB
{
    Vec3 minPoint, maxPoint;

public:
    static AABB combineAABBs(const AABB &b1, const AABB &b2);
    AABB(const Vec3 &minPoint, const Vec3 &maxPoint);
    ~AABB() noexcept = default;

    const Vec3 &getMinPoint() const;
    const Vec3 &getMaxPoint() const;

    bool passesThrough(const Ray &ray, double tmin, double tmax) const;
};