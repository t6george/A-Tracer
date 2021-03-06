#pragma once
#include <Hittable.cuh>
#include <AABB.cuh>

template <utils::Axis A>
class AARotate : public Hittable
{
    const SharedPointer<Hittable> shape;
    double sinTheta, cosTheta;
    AABB bbox;

    DEV AABB computeBoundingBox();
    DEV void rotateCoords(Vec3 &v, const double sin) const;
    DEV void inverseRotateCoords(Vec3 &v, const double sin) const;

public:
    DEV HOST AARotate(const SharedPointer<Hittable> shape, double angle);
    DEV HOST ~AARotate() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};
