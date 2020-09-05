#include <ConstantVolume.cuh>
#include <IsotropicMaterial.cuh>
#include <Texture.cuh>

DEV HOST ConstantVolume::ConstantVolume(const SharedPointer<Hittable> boundary,
                               const SharedPointer<Texture> phaseFunction,
                               const double density)
    : boundary{boundary}, phaseFunction{std::make_shared<IsotropicMaterial>(phaseFunction)}, densityReciprocal{-1. / density} {}

DEV Hittable::HitType ConstantVolume::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    HitType hit = HitType::NO_HIT;

    HitRecord rec1, rec2;

    if (static_cast<bool>(boundary->getCollisionData(ray, rec1, -utils::infinity, utils::infinity, flip)) &&
        static_cast<bool>(boundary->getCollisionData(ray, rec2, rec1.t + .0001, utils::infinity, flip)))
    {
        rec1.t = fmax(tMin, rec1.t);
        rec2.t = fmin(tMax, rec2.t);

        if (rec1.t < rec2.t)
        {
            rec1.t = fmax(0., rec1.t);
            const double rayLen = ray.getDirection().len();
            const double innerDist = rayLen * (rec2.t - rec1.t);
            const double outerDist = log(utils::random_double()) * densityReciprocal;

            if (outerDist <= innerDist)
            {
                record.t = rec1.t + outerDist / rayLen;
                record.point = ray.eval(record.t);

                record.normal = Vec3{1., 0., 0.};
                record.isInFront = true;
                hit = phaseFunction->scatterRay(ray, record) ? HitType::HIT_SCATTER
                                                             : HitType::HIT_NO_SCATTER;
            }
        }
    }

    return hit;
}

DEV bool ConstantVolume::getBoundingBox(double time0, double time1, AABB &box) const
{
    return boundary->getBoundingBox(time0, time1, box);
}