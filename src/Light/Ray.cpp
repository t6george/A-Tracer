#include <Ray.cuh>

Ray::Ray() : orig{Vec3{}}, dir{Vec3{}}, tm{0.} {}

Ray::Ray(const Vec3 &origin, const Vec3 &direction, double time)
    : orig{origin}, dir{direction}, tm{time} {}

double Ray::getTime() const { return tm; }

void Ray::setTime(double time) { tm = time; }

const Vec3 &Ray::getOrigin() const { return orig; }

const Vec3 &Ray::getDirection() const { return dir; }

void Ray::setOrigin(const Vec3 &otherV) { orig = otherV; }

void Ray::setDirection(const Vec3 &otherV) { dir = otherV; }

Vec3 Ray::eval(double t) const { return orig + (dir * t); }
