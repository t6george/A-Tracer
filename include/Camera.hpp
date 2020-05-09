#pragma once
#include <Vec3.hpp>
#include <Ray.hpp>

class Camera
{
    // make this const?
    Vec3 corner, dimX, dimY, origin;

    Ray lineOfSight;

public:
    Camera(const Vec3 &corner, const Vec3 &dimX, const Vec3 &dimY, const Vec3 &origin);
    ~Camera() noexcept = default;
    const Ray &updateLineOfSight(double u, double v);
    const Ray &getLineOfSight() const;
    void moveCamera(const Vec3 &displacement);
};