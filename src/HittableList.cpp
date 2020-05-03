#include <HittableList.hpp>

bool HittableList::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax) const
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

void HittableList::add(std::shared_ptr<Hittable> hittable) { hittables.emplace_back(hittable); }

void HittableList::clear() { hittables.clear(); }