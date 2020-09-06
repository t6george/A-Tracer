#pragma once
#include <Memory.cuh>

#include <Hittable.cuh>
#include <AABB.cuh>

class HittableList;

class BVHNode : public Hittable
{
    SharedPointer<Hittable> left, right;
    AABB boundingBox;

public:
    HOST BVHNode() = default;
    HOST BVHNode(HittableList &world, const double time0, const double time1);
    HOST BVHNode(HittableList &world, const double time0, const double time1, const size_t start, const size_t end);

    HOST ~BVHNode() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};

template <utils::Axis dim>
DEV inline bool dimCompare(const SharedPointer<Hittable> a, const SharedPointer<Hittable> b)
{
    AABB bbA, bbB;
    a->getBoundingBox(0., 0., bbA);
    b->getBoundingBox(0., 0., bbB);

    return bbA.getMinPoint()[static_cast<int>(dim)] < bbB.getMinPoint()[static_cast<int>(dim)];
}

DEV inline bool xCompare(const SharedPointer<Hittable> a, const SharedPointer<Hittable> b)
{
    return dimCompare<utils::Axis::X>(a, b);
}

DEV inline bool yCompare(const SharedPointer<Hittable> a, const SharedPointer<Hittable> b)
{
    return dimCompare<utils::Axis::Y>(a, b);
}

DEV inline bool zCompare(const SharedPointer<Hittable> a, const SharedPointer<Hittable> b)
{
    return dimCompare<utils::Axis::Z>(a, b);
}
