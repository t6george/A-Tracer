#include <Ray.hpp>

Ray::Ray() : orig{Vec3{}}, dir{Vec3{}} {}

Ray::Ray(const Vec3 &origin) : orig{origin}, dir{Vec3{}} {}

Ray::Ray(const Vec3 &origin, const Vec3 &direction) : orig{origin}, dir{direction} {}

const Vec3 &Ray::origin() const { return orig; }

const Vec3 &Ray::direction() const { return dir; }

void Ray::resetOrigin(const Vec3 &otherV) { orig = otherV; }

void Ray::resetDirection(const Vec3 &otherV) { dir = otherV; }

Vec3 Ray::eval(double t) const { return orig + (dir * t); }
