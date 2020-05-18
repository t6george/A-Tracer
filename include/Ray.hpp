#pragma once
#include <Vec3.hpp>

class Ray
{
    Vec3 orig, dir;
    double tm;

public:
    Ray();
    Ray(const Vec3 &origin, const Vec3 &direction = Vec3{}, double time = 0.);
    ~Ray() noexcept = default;

    const Vec3 &origin() const;
    const Vec3 &direction() const;
    double time() const;
    void setTime(double time);

    void resetOrigin(const Vec3 &otherV);
    void resetDirection(const Vec3 &otherV);

    Vec3 eval(double t) const;
};