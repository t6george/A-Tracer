#pragma once
#include <Vec3.hpp>

class Ray;

class AABB
{
    Vec3 minPoint, maxPoint;

public:
    static AABB combineAABBs(const AABB &b1, const AABB &b2);
    AABB(const Vec3 &minPoint = Vec3{}, const Vec3 &maxPoint = Vec3{});
    ~AABB() noexcept = default;

    const Vec3 &getMinPoint() const;
    const Vec3 &getMaxPoint() const;

    void setMinPoint(const Vec3 &v);
    void setMaxPoint(const Vec3 &v);

    bool passesThrough(const Ray &ray, double tmin, double tmax) const;
};