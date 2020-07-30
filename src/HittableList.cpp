#include <HittableList.hpp>
#include <AABB.hpp>

Hittable::HitType HittableList::getCollisionData(const Ray &ray, Hittable::HitRecord &record, double tMin, double tMax, bool flip)
{
    Hittable::HitRecord tmpRecord;
    Hittable::HitType collisionType = Hittable::HitType::NO_HIT, tmpCollisionType;

    for (const auto &obj : hittables)
    {
        if (static_cast<bool>(tmpCollisionType = obj->getCollisionData(ray, tmpRecord, tMin, tMax, flip)))
        {
            collisionType = tmpCollisionType;
            record = tmpRecord;
            tMax = record.t;
        }
    }

    return collisionType;
}

bool HittableList::getBoundingBox(double time0, double time1, AABB &box) const
{
    bool firstBox = true;
    AABB tmp, outputBox;

    for (const auto &obj : hittables)
    {
        if (!obj->getBoundingBox(time0, time1, tmp))
            return false;
        outputBox = firstBox ? tmp : AABB::combineAABBs(outputBox, tmp);
        firstBox = false;
    }

    return !hittables.empty();
}

void HittableList::add(std::shared_ptr<Hittable> hittable) { hittables.emplace_back(hittable); }

void HittableList::clear() { hittables.clear(); }