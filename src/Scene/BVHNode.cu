#include <algorithm>
#include <cassert>
#include <iostream>

#include <BVHNode.cuh>
#include <Util.cuh>
#include <HittableList.cuh>
#include <WeightedPdf.cuh>

DEV HOST BVHNode::BVHNode(HittableList &world, const double time0, const double time1)
    : BVHNode{world, time0, time1, 0, world.hittables.size()} { assert(world.hittables.size() > 0); }

DEV HOST BVHNode::BVHNode(HittableList &world, const double time0, const double time1, const size_t start, const size_t end)
{
    size_t span = end - start;

    int axis = utils::random_int(0, 3);
    auto cmp = axis == 0 ? xCompare : axis == 1 ? yCompare : zCompare;

    switch (span)
    {
    case 1:
        left = right = world.hittables[start];
        break;
    case 2:
        left = world.hittables[start];
        right = world.hittables[start + 1];
        if (!cmp(world.hittables[start], world.hittables[start + 1]))
        {
            utils::swap(left, right);
        }
        break;
    default:
        size_t mid = start + span / 2;
        std::sort(world.hittables.begin() + start, world.hittables.begin() + end, cmp);

        left = std::make_shared<BVHNode>(world, time0, time1, start, mid);
        right = std::make_shared<BVHNode>(world, time0, time1, mid, end);
        break;
    }

    AABB leftBox, rightBox;

    left->getBoundingBox(time0, time1, leftBox);
    right->getBoundingBox(time0, time1, rightBox);

    boundingBox = AABB::combineAABBs(leftBox, rightBox);
}

DEV Hittable::HitType BVHNode::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    if (!boundingBox.passesThrough(ray, tMin, tMax))
    {
        return Hittable::HitType::NO_HIT;
    }

    Hittable::HitType hitLeft = left->getCollisionData(ray, record, tMin, tMax, flip);
    Hittable::HitType hitRight = right->getCollisionData(ray, record, tMin, static_cast<bool>(hitLeft) ? record.t : tMax, flip);

    return static_cast<bool>(hitRight) ? hitRight : hitLeft;
}

DEV bool BVHNode::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}
