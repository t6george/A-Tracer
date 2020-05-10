#include <Camera.hpp>
#include <Utils.hpp>

Camera::Camera(const double aspR, double fov, const Vec3 &origin)
    : lineOfSight{Ray{origin}}, halfHeight{tan(deg_to_rad(fov / 2.))},
      halfWidth{aspR * halfHeight}, origin{origin},
      corner{Vec3{-halfWidth, -halfHeight, -1.}},
      dimX{Vec3{2. * halfWidth, 0., 0.}}, dimY{Vec3{0., 2. * halfHeight, 0.}}
{
}

const Ray &Camera::updateLineOfSight(double u, double v)
{
    lineOfSight.resetDirection(corner + dimX * u + dimY * v - origin);
    return lineOfSight;
}

const Ray &Camera::getLineOfSight() const
{
    return lineOfSight;
}

void Camera::moveCamera(const Vec3 &displacement)
{
    origin += displacement;
    lineOfSight = origin;
}