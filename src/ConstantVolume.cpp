#include <ConstantVolume.hpp>
#include <Material.hpp>

ConstantVolume::ConstantVolume(const std::shared_ptr<Hittable> boundary,
                               const std::shared_ptr<Material> phaseFunction,
                               const double density)
    : boundary{boundary}, phaseFunction{phaseFunction}, densityReciprocal{-1. / density} {}

Hittable::HitType ConstantVolume::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    HitType hit = HitType::NO_HIT;

    HitRecord rec1, rec2;

    if (static_cast<bool>(boundary->getCollisionData(ray, rec1, -utils::infinity, utils::infinity)) &&
        static_cast<bool>(boundary->getCollisionData(ray, rec2, rec1.t + .0001, utils::infinity)))
    {
        rec1.t = fmax(tMin, rec1.t);
        rec2.t = fmin(tMax, rec2.t);

        if (rec1.t < rec2.t)
        {
            rec1.t = fmax(0., rec1.t);
            const double rayLen = ray.getDirection().len();
            const double innerDist = rayLen * (rec2.t - rec1.t);
            const double outerDist = log(utils::random_double()) * densityReciprocal;

            if (outerDist > innerDist)
            {
                record.t = rec1.t + outerDist / rayLen;
                record.point = ray.eval(record.t);

                record.normal = Vec3{1., 0., 0.};
                record.isInFront = true;
                hit = phaseFunction->scatterRay(ray, record) ? HitType::HIT_SCATTER : HitType::HIT_NO_SCATTER;
            }
        }
    }

    return hit;
}

bool ConstantVolume::getBoundingBox(double time0, double time1, AABB &box) const
{
    return boundary->getBoundingBox(time0, time1, box);
}