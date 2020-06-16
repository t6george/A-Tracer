#include <FlipFace.hpp>

FlipFace::FlipFace(const std::shared_ptr<Hittable> hittable) : hittable{hittable} {}

Hittable::HitType FlipFace::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    Hittable::HitType isHit;
    record.flip = true;
    isHit = hittable->getCollisionData(ray, record, tMin, tMax);
    record.flip = false;
    // if (static_cast<bool>(isHit = hittable->getCollisionData(ray, record, tMin, tMax)))
    // {
    //     record.isInFront ^= true;
    // }
    return isHit;
}

bool FlipFace::getBoundingBox(double time0, double time1, AABB &box) const
{
    return hittable->getBoundingBox(time0, time1, box);
}