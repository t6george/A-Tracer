#include <cmath>

#include <AABB.hpp>
#include <Ray.hpp>

AABB::AABB(const Vec3 &minPoint, const Vec3 &maxPoint)
    : minPoint{minPoint}, maxPoint{maxPoint} {}

const Vec3 &AABB::getMinPoint() const { return minPoint; }

const Vec3 &AABB::getMaxPoint() const { return maxPoint; }

bool AABB::passesThrough(const Ray &ray, double tmin, double tmax) const
{
    bool isHit = true;
    double t0, t1;

    for (int a = 0; a < 3; ++a)
    {
        t0 = (minPoint[a] - ray.origin()[a]) / ray.direction()[a];
        t0 = (maxPoint[a] - ray.origin()[a]) / ray.direction()[a];

        if (t0 > t1)
        {
            std::swap(t0, t1);
        }

        tmin = fmax(t0, tmin);
        tmax = fmin(t1, tmax);
    }

    return isHit;
}
