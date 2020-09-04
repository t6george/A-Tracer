#include <Ray.cuh>

DEV HOST Ray::Ray() : orig{Vec3{}}, dir{Vec3{}}, tm{0.} {}

DEV HOST Ray::Ray(const Vec3 &origin, const Vec3 &direction, double time)
    : orig{origin}, dir{direction}, tm{time} {}

DEV double Ray::getTime() const { return tm; }

DEV void Ray::setTime(double time) { tm = time; }

DEV const Vec3 &Ray::getOrigin() const { return orig; }

DEV const Vec3 &Ray::getDirection() const { return dir; }

DEV void Ray::setOrigin(const Vec3 &otherV) { orig = otherV; }

DEV void Ray::setDirection(const Vec3 &otherV) { dir = otherV; }

DEV Vec3 Ray::eval(double t) const { return orig + (dir * t); }
