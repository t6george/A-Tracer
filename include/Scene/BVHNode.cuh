#pragma once
#include <memory>

#include <Hittable.cuh>
#include <AABB.cuh>

class HittableList;

class BVHNode : public Hittable
{
    std::shared_ptr<Hittable> left, right;
    AABB boundingBox;

public:
    DEV HOST BVHNode() = default;
    DEV HOST BVHNode(HittableList &world, const double time0, const double time1);
    DEV HOST BVHNode(HittableList &world, const double time0, const double time1, const size_t start, const size_t end);

    DEV HOST ~BVHNode() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity, bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;
};

template <utils::Axis dim>
inline DEV bool dimCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    AABB bbA, bbB;
    a->getBoundingBox(0., 0., bbA);
    b->getBoundingBox(0., 0., bbB);

    return bbA.getMinPoint()[static_cast<int>(dim)] < bbB.getMinPoint()[static_cast<int>(dim)];
}

inline DEV bool xCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<utils::Axis::X>(a, b);
}

inline DEV bool yCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<utils::Axis::Y>(a, b);
}

inline DEV bool zCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<utils::Axis::Z>(a, b);
}