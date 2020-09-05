#include <AARotate.cuh>

template <enum utils::Axis A>
DEV HOST AARotate<A>::AARotate(const SharedPointer<Hittable> shape, double angle)
    : shape{shape}, sinTheta{sin(utils::deg_to_rad(angle))}, cosTheta{cos(utils::deg_to_rad(angle))},
      bbox{computeBoundingBox()} {}

template <enum utils::Axis A>
DEV AABB AARotate<A>::computeBoundingBox()
{
    AABB box;
    Vec3 minPoint{utils::infinity, utils::infinity, utils::infinity};
    Vec3 maxPoint{-utils::infinity, -utils::infinity, -utils::infinity};

    if (shape->getBoundingBox(0., 1., box))
    {
        Vec3 candidateExtreme;

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                for (int k = 0; k < 2; ++k)
                {
                    candidateExtreme[0] = i * box.getMaxPoint().x() + (1 - i) * box.getMinPoint().x();
                    candidateExtreme[1] = j * box.getMaxPoint().y() + (1 - j) * box.getMinPoint().y();
                    candidateExtreme[2] = k * box.getMaxPoint().z() + (1 - k) * box.getMinPoint().z();

                    rotateCoords(candidateExtreme, sinTheta);

                    for (int l = 0; l < 3; ++l)
                    {
                        minPoint[l] = fmin(candidateExtreme[l], minPoint[l]);
                        maxPoint[l] = fmax(candidateExtreme[l], maxPoint[l]);
                    }
                }
            }
        }
    }

    return AABB{minPoint, maxPoint};
}

template <enum utils::Axis A>
DEV Hittable::HitType AARotate<A>::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    Hittable::HitType hit;
    Vec3 origin = ray.getOrigin();
    Vec3 direction = ray.getDirection();

    rotateCoords(origin, sinTheta);
    rotateCoords(direction, sinTheta);

    Ray adjusted{origin, direction, ray.getTime()};

    if (static_cast<bool>(hit = shape->getCollisionData(adjusted, record, tMin, tMax, flip)))
    {
        Vec3 scatteredOrigin = record.scatteredRay.getOrigin();
        Vec3 scatteredDirection = record.scatteredRay.getDirection();
        inverseRotateCoords(scatteredOrigin, sinTheta);
        inverseRotateCoords(record.normal, sinTheta);

        record.scatteredRay.setOrigin(scatteredOrigin);
        record.scatteredRay.setDirection(scatteredDirection);
        record.setLightPosition(adjusted);
    }

    return hit;
}

template <enum utils::Axis A>
DEV bool AARotate<A>::getBoundingBox(double time0, double time1, AABB &box) const
{
    AABB tmp;
    bool hasBox = shape->getBoundingBox(time0, time1, tmp);

    if (hasBox)
    {
        box = bbox;
    }

    return hasBox;
}

template <>
DEV void AARotate<utils::Axis::X>::rotateCoords(Vec3 &v, const double sin) const
{
    v = Vec3{
        v.x(),
        cosTheta * v.y() - sin * v.z(),
        cosTheta * v.z() + sin * v.y()};
}

template <>
DEV void AARotate<utils::Axis::Y>::rotateCoords(Vec3 &v, const double sin) const
{
    v = Vec3{
        cosTheta * v.x() - sin * v.z(),
        v.y(),
        cosTheta * v.z() + sin * v.x()};
}

template <>
DEV void AARotate<utils::Axis::Z>::rotateCoords(Vec3 &v, const double sin) const
{
    v = Vec3{
        cosTheta * v.x() - sin * v.y(),
        cosTheta * v.y() + sin * v.x(),
        v.z()};
}

template <enum utils::Axis A>
DEV void AARotate<A>::inverseRotateCoords(Vec3 &v, const double sin) const
{
    rotateCoords(v, -sin);
}

template class AARotate<utils::Axis::X>;
template class AARotate<utils::Axis::Y>;
template class AARotate<utils::Axis::Z>;