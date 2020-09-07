#include <FlipFace.cuh>

DEV HOST FlipFace::FlipFace(const SharedPointer<Hittable> hittable) : hittable{hittable} {}

DEV Hittable::HitType FlipFace::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    return hittable->getCollisionData(ray, record, tMin, tMax, true);
}

DEV bool FlipFace::getBoundingBox(double time0, double time1, AABB &box) const
{
    return hittable->getBoundingBox(time0, time1, box);
}
