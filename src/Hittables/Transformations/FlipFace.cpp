#include <FlipFace.cuh>

FlipFace::FlipFace(const std::shared_ptr<Hittable> hittable) : hittable{hittable} {}

Hittable::HitType FlipFace::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    return hittable->getCollisionData(ray, record, tMin, tMax, true);
}

bool FlipFace::getBoundingBox(double time0, double time1, AABB &box) const
{
    return hittable->getBoundingBox(time0, time1, box);
}