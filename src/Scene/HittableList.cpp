#include <cassert>

#include <HittableList.cuh>
#include <AABB.cuh>

DEV DEV Hittable::HitType HittableList::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
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

DEV bool HittableList::getBoundingBox(double time0, double time1, AABB &box) const
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

DEV void HittableList::add(std::shared_ptr<Hittable> hittable) { hittables.emplace_back(hittable); }

DEV void HittableList::clear() { hittables.clear(); }

DEV Vec3 HittableList::genRandomVector(const Vec3& origin) const
{
    return hittables.at(utils::random_int(0, hittables.size()))->genRandomVector(origin);
}

DEV double HittableList::eval(const Vec3& origin, const Vec3& v, bool flip) const
{
    assert(hittables.size() > 0);
    double weight = 1. / hittables.size();
    double sum = 0.;

    for (const auto& hittable : hittables)
    {
        sum += weight * hittable->eval(origin, v, flip);
    }

    return sum;
}
