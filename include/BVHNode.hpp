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
    enum Dim3D
    {
        X,
        Y,
        Z
    };

    BVHNode() = default;
    BVHNode(HittableList &world, const size_t start, const size_t end, const double time0, const double time1);

    ~BVHNode() noexcept = default;

    bool getCollisionData(const Ray &ray, HitRecord &record,
                          double tMin = -infinity,
                          double tMax = infinity) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};

template <BVHNode::Dim3D dim>
inline bool dimCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    AABB bbA, bbB;
    a->getBoundingBox(0., 0., bbA);
    b->getBoundingBox(0., 0., bbB);

    return bbA.getMinPoint()[dim] < bbB.getMinPoint()[dim];
}

inline bool xCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<BVHNode::Dim3D::X>(a, b);
}

inline bool yCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<BVHNode::Dim3D::Y>(a, b);
}

inline bool zCompare(const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b)
{
    return dimCompare<BVHNode::Dim3D::Z>(a, b);
}