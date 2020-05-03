#pragma once
#include <Vec3.hpp>

class Ray
{
    Vec3 orig, dir;

public:
    Ray();
    Ray(const Vec3 &origin, const Vec3 &direction = Vec3{});
    ~Ray() noexcept = default;

    const Vec3 &origin() const;
    const Vec3 &direction() const;

    void resetOrigin(const Vec3 &otherV);
    void resetDirection(const Vec3 &otherV);

    Vec3 eval(double t) const;
};