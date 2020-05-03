#include <Camera.hpp>

Camera::Camera(const Vec3 &corner, const Vec3 &dimX, const Vec3 &dimY, const Vec3 &origin)
    : corner{corner}, dimX{dimX}, dimY{dimY}, origin{origin}, lineOfSight{Ray{origin}} {}

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