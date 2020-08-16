#pragma once

#include <Vec3.hpp>
#include <Macro.hpp>

class Ray
{
    Vec3 orig, dir;
    double tm;

public:
    DEV HOST Ray();
    DEV HOST Ray(const Vec3 &origin, const Vec3 &direction = Vec3{}, double time = 0.);
    DEV HOST ~Ray() noexcept = default;

    DEV double getTime() const;
    DEV void setTime(double time);

    DEV const Vec3 &getOrigin() const;
    DEV const Vec3 &getDirection() const;
    DEV void setOrigin(const Vec3 &otherV);
    DEV void setDirection(const Vec3 &otherV);

    DEV Vec3 eval(double t) const;
};