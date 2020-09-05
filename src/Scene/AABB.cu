#include <cmath>

#include <AABB.cuh>
#include <Ray.cuh>

DEV HOST AABB AABB::combineAABBs(const AABB &b1, const AABB &b2)
{
    Vec3 minPoint{
        fmin(b1.getMinPoint().x(), b2.getMinPoint().x()),
        fmin(b1.getMinPoint().y(), b2.getMinPoint().y()),
        fmin(b1.getMinPoint().z(), b2.getMinPoint().z())};

    Vec3 maxPoint{
        fmax(b1.getMaxPoint().x(), b2.getMaxPoint().x()),
        fmax(b1.getMaxPoint().y(), b2.getMaxPoint().y()),
        fmax(b1.getMaxPoint().z(), b2.getMaxPoint().z())};

    return AABB{minPoint, maxPoint};
}

DEV HOST AABB::AABB(const Vec3 &minPoint, const Vec3 &maxPoint)
    : minPoint{minPoint}, maxPoint{maxPoint} {}

DEV const Vec3 &AABB::getMinPoint() const { return minPoint; }

DEV const Vec3 &AABB::getMaxPoint() const { return maxPoint; }

DEV void AABB::setMinPoint(const Vec3 &v) { minPoint = v; }

DEV void AABB::setMaxPoint(const Vec3 &v) { maxPoint = v; }

DEV bool AABB::passesThrough(const Ray &ray, double tmin, double tmax) const
{
    double invD, t0, t1;

    for (int a = 0; a < 3; ++a)
    {
        invD = 1. / ray.getDirection()[a];
        t0 = (minPoint[a] - ray.getOrigin()[a]) * invD;
        t1 = (maxPoint[a] - ray.getOrigin()[a]) * invD;

        if (invD < 0.)
        {
            utils::swap(t0, t1);
        }

        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmin > tmax)
        {
            return false;
        }
    }

    return true;
}
