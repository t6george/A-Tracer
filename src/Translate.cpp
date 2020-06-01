#include <Translate.hpp>

Translate::Translate(const std::shared_ptr<Shape> shape, const Vec3 &displacement)
    : shape{shape}, displacement{displacement} {}

Hittable::HitType Translate::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    Hittable::HitType hit;
    Ray moved{ray.origin() - displacement, ray.direction(), ray.getTime()};

    if (static_cast<bool>(hit = shape->getCollisionData(moved, record, tMin, tMax)))
    {
    }

    return hit;
}

bool Translate::getBoundingBox(double time0, double time1, AABB &box) const
{
    bool hasBox;
    if (shape->getBoundingBox(time0, time1, box))
    {
        hasBox = true;
        box.setMinPoint(box.getMinPoint() + displacement);
        box.setMaxPoint(box.getMaxPoint() + displacement);
    }
    else
    {
        hasBox = false;
    }

    return hasBox;
}
