#include <Translate.cuh>
#include <AABB.cuh>

DEV HOST Translate::Translate(const SharedPointer<Hittable> shape, const Vec3 &displacement)
    : shape{shape}, displacement{displacement} {}

DEV Hittable::HitType Translate::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    Hittable::HitType hit;
    Ray moved{ray.getOrigin() - displacement, ray.getDirection(), ray.getTime()};

    if (static_cast<bool>(hit = shape->getCollisionData(moved, record, tMin, tMax, flip)))
    {
        record.point += displacement;
        record.scatteredRay.setOrigin(record.scatteredRay.getOrigin() + displacement);
        record.setLightPosition(moved);
    }

    return hit;
}

DEV bool Translate::getBoundingBox(double time0, double time1, AABB &box) const
{
    bool hasBox = false;
    if (shape->getBoundingBox(time0, time1, box))
    {
        hasBox = true;
        box = AABB{box.getMinPoint() + displacement, box.getMaxPoint() + displacement};
    }

    return hasBox;
}
