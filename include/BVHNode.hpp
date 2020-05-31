#pragma once
#include <memory>

#include <Hittable.hpp>
#include <AABB.hpp>

class HittableList;

class BVHNode : public Hittable
{
    std::shared_ptr<Hittable> left, right;
    AABB boundingBox;

public:
    BVHNode() = default;
    BVHNode(HittableList &world, const size_t start, const size_t end, const double time0, const double time1);

    ~BVHNode() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};

#include <AARect.hpp>

template <Axis dim>
inline bool dimCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    AABB bbA, bbB;
    a->getBoundingBox(0., 0., bbA);
    b->getBoundingBox(0., 0., bbB);

    return bbA.getMinPoint()[static_cast<int>(dim)] < bbB.getMinPoint()[static_cast<int>(dim)];
}

inline bool xCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<Axis::X>(a, b);
}

inline bool yCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<Axis::Y>(a, b);
}

inline bool zCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<Axis::Z>(a, b);
}