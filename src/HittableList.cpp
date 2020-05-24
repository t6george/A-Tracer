#include <HittableList.hpp>
#include <AABB.hpp>

bool HittableList::getCollisionData(const Ray &ray, Hittable::HitRecord &record, double tMin, double tMax)
{
    Hittable::HitRecord tmpRecord;
    bool isCollision = false;

    for (const auto &obj : hittables)
    {
        if (obj->getCollisionData(ray, tmpRecord, tMin, tMax))
        {
            isCollision = true;
            record = tmpRecord;
            tMax = record.t;
        }
    }

    return isCollision;
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