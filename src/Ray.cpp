#include <Ray.hpp>

Ray::Ray() : orig{Vec3{}}, dir{Vec3{}}, tm{0.} {}

Ray::Ray(const Vec3 &origin, const Vec3 &direction, double time)
    : orig{origin}, dir{direction}, tm{time} {}

const Vec3 &Ray::origin() const { return orig; }

const Vec3 &Ray::direction() const { return dir; }

double Ray::getTime() const { return tm; }

void Ray::setTime(double time) { tm = time; }

void Ray::resetOrigin(const Vec3 &otherV) { orig = otherV; }

void Ray::resetDirection(const Vec3 &otherV) { dir = otherV; }

Vec3 Ray::eval(double t) const { return orig + (dir * t); }
