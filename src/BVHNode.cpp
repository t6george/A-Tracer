#include <algorithm>

#include <BVHNode.hpp>
#include <Utils.hpp>
#include <HittableList.hpp>

BVHNode::BVHNode(HittableList &world, const size_t start, const size_t end, const double time0, const double time1)
{
    size_t span = end - start;

    int axis = random_int(0, 3);
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
            std::swap(left, right);
        }
        break;
    default:
        size_t mid = start + span / 2;
        std::sort(world.hittables.begin() + start, world.hittables.begin() + end, cmp);

        left = std::make_shared<BVHNode>(world, start, mid, time0, time1);
        right = std::make_shared<BVHNode>(world, mid, end, time0, time1);
        break;
    }

    AABB leftBox, rightBox;

    left->getBoundingBox(time0, time1, leftBox);
    right->getBoundingBox(time0, time1, rightBox);

    boundingBox = AABB::combineAABBs(leftBox, rightBox);
}

bool BVHNode::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    if (!boundingBox.passesThrough(ray, tMin, tMax))
    {
        return false;
    }

    bool hitLeft = left->getCollisionData(ray, record, tMin, tMax);
    bool hitRight = right->getCollisionData(ray, record, tMin, hitLeft ? record.t : tMax);

    return hitLeft || hitRight;
}

bool BVHNode::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}