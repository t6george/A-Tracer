#include <Camera.cuh>
#include <Util.cuh>

HOST Camera::OrthoNormalBasis::OrthoNormalBasis(Vec3 &eyes, const Vec3 &lookat, const Vec3 &vup)
    : z{(eyes - lookat).getUnitVector()}, x{vup.x(z).getUnitVector()}, y{z.x(x)} {}

HOST Camera::Camera(const double aspR, double fov, const double aperture, const double focusD,
               const Vec3 &lookfrom, const Vec3 &lookat, double t1, double t2)
    : lineOfSight{Ray{lookfrom}}, halfHeight{tan(utils::deg_to_rad(fov / 2.))}, halfWidth{aspR * halfHeight},
      lensRadius{aperture / 2.}, focusDist{focusD}, eyes{lookfrom}, basis{Camera::OrthoNormalBasis{eyes, lookat}},
      corner{eyes - basis.getX() * halfWidth * focusDist - basis.getY() * halfHeight * focusDist - basis.getZ() * focusDist},
      dimX{2. * halfWidth * focusDist * basis.getX()}, dimY{2. * halfHeight * focusDist * basis.getY()},
      time1{t1}, time2{t2} {}

DEV const Vec3 &Camera::OrthoNormalBasis::getX() const { return x; }

DEV const Vec3 &Camera::OrthoNormalBasis::getY() const { return y; }

DEV const Vec3 &Camera::OrthoNormalBasis::getZ() const { return z; }

DEV Ray &Camera::updateLineOfSight(double u, double v)
{
    Vec3 rd = Vec3::randomUnitCircleVec() * lensRadius;
    Vec3 offset = basis.getX() * rd.x() + basis.getY() * rd.y();
    lineOfSight.setOrigin(eyes + offset);
    lineOfSight.setDirection(corner + dimX * u + dimY * v - eyes - offset);
    lineOfSight.setTime(utils::random_double(time1, time2));
    return lineOfSight;
}

DEV const Ray &Camera::getLineOfSight() const
{
    return lineOfSight;
}

DEV void Camera::moveCamera(const Vec3 &displacement)
{
    eyes += displacement;
    lineOfSight.setOrigin(eyes);
}
