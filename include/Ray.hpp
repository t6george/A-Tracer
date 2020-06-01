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

    double getTime() const;
    void setTime(double time);

    const Vec3 &getOrigin() const;
    const Vec3 &getDirection() const;
    void setOrigin(const Vec3 &otherV);
    void setDirection(const Vec3 &otherV);

    Vec3 eval(double t) const;
};