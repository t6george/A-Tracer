#include <Camera.hpp>
#include <Utils.hpp>

Camera::OrthoNormalBasis::OrthoNormalBasis(Vec3 &eyes, const Vec3 &lookat, const Vec3 &vup)
    : z{(eyes - lookat).getUnitVector()}, x{vup.x(z).getUnitVector()}, y{z.x(x)} {}

const Vec3 &Camera::OrthoNormalBasis::getX() const { return x; }

const Vec3 &Camera::OrthoNormalBasis::getY() const { return y; }

const Vec3 &Camera::OrthoNormalBasis::getZ() const { return z; }

Camera::Camera(const double aspR, double fov, const Vec3 &lookfrom, const Vec3 &lookat)
    : lineOfSight{Ray{lookfrom}}, halfHeight{tan(deg_to_rad(fov / 2.))},
      halfWidth{aspR * halfHeight}, eyes{lookfrom}, basis{Camera::OrthoNormalBasis{eyes, lookat}},
      corner{eyes - basis.getX() * halfWidth - basis.getY() * halfHeight - basis.getZ()},
      dimX{2. * halfWidth * basis.getX()}, dimY{2. * halfHeight * basis.getY()} {}

const Ray &Camera::updateLineOfSight(double u, double v)
{
    lineOfSight.resetDirection(corner + dimX * u + dimY * v - eyes);
    return lineOfSight;
}

const Ray &Camera::getLineOfSight() const
{
    return lineOfSight;
}

void Camera::moveCamera(const Vec3 &displacement)
{
    eyes += displacement;
    lineOfSight = eyes;
}