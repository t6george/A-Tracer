#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>
#include <Ray.cuh>

class Camera
{
    class OrthoNormalBasis
    {
        Vec3 z, x, y;

    public:
        HOST OrthoNormalBasis(Vec3 &eyes, const Vec3 &lookat, const Vec3 &vup = Vec3{0., 1., 0.});
        HOST ~OrthoNormalBasis() noexcept = default;

        DEV const Vec3 &getX() const;
        DEV const Vec3 &getY() const;
        DEV const Vec3 &getZ() const;
    };

    Ray lineOfSight;

    const double halfHeight, halfWidth;
    double lensRadius, focusDist;

    Vec3 eyes;

    OrthoNormalBasis basis;

    Vec3 corner, dimX, dimY;
    double time1, time2;

public:
    HOST Camera(const double aspR, double fov, const double aperture, const double focusD,
           const Vec3 &lookfrom = Vec3{}, const Vec3 &lookat = Vec3{0., 0., -1.},
           double t0 = 0., double t1 = 0.);

    HOST ~Camera() noexcept = default;
    
    DEV Ray &updateLineOfSight(double u, double v);
    DEV const Ray &getLineOfSight() const;
    DEV void moveCamera(const Vec3 &displacement);
};
