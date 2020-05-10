#pragma once
#include <Vec3.hpp>
#include <Ray.hpp>

class Camera
{
    Ray lineOfSight;

    const double halfHeight, halfWidth;

    Vec3 origin, corner, dimX, dimY;

public:
    Camera(const double aspR, double fov, const Vec3 &origin = Vec3{});
    ~Camera() noexcept = default;
    const Ray &updateLineOfSight(double u, double v);
    const Ray &getLineOfSight() const;
    void moveCamera(const Vec3 &displacement);
};