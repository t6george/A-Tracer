#include <AARotate.hpp>

template <enum utils::Axis A>
AARotate<A>::AARotate(const std::shared_ptr<Hittable> shape, double angle)
    : shape{shape}, sinTheta{sin(utils::deg_to_rad(angle))}, cosTheta{cos(utils::deg_to_rad(angle))},
      bbox{computeBoundingBox()} {}

template <enum utils::Axis A>
AABB AARotate<A>::computeBoundingBox()
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

                    rotateCoords(candidateExtreme);

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
Hittable::HitType AARotate<A>::getCollisionData(const Ray &ray, Hittable::HitRecord &record, double tMin, double tMax, bool flip)
{
    Hittable::HitType hit;
    Vec3 origin = ray.getOrigin();
    Vec3 direction = ray.getDirection();

    rotateCoords(origin);
    rotateCoords(direction);

    Ray adjusted{origin, direction, ray.getTime()};

    if (static_cast<bool>(hit = shape->getCollisionData(adjusted, record, tMin, tMax, flip)))
    {
        Vec3 scatteredOrigin = record.scatteredRay.getOrigin();
        Vec3 scatteredDirection = record.scatteredRay.getDirection();
        inverseRotateCoords(scatteredOrigin);
        inverseRotateCoords(scatteredDirection);

        record.scatteredRay.setOrigin(scatteredOrigin);
        record.scatteredRay.setDirection(scatteredDirection);
    }

    return hit;
}

template <enum utils::Axis A>
bool AARotate<A>::getBoundingBox(double time0, double time1, AABB &box) const
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
void AARotate<utils::Axis::X>::rotateCoords(Vec3 &v) const
{
    v = Vec3{
        v.x(),
        cosTheta * v.y() - sinTheta * v.z(),
        cosTheta * v.z() + sinTheta * v.y()};
}

template <>
void AARotate<utils::Axis::Y>::rotateCoords(Vec3 &v) const
{
    v = Vec3{
        cosTheta * v.x() - sinTheta * v.z(),
        v.y(),
        cosTheta * v.z() + sinTheta * v.x()};
}

template <>
void AARotate<utils::Axis::Z>::rotateCoords(Vec3 &v) const
{
    v = Vec3{
        cosTheta * v.x() - sinTheta * v.y(),
        cosTheta * v.y() + sinTheta * v.x(),
        v.z()};
}

template <enum utils::Axis A>
void AARotate<A>::inverseRotateCoords(Vec3 &v)
{
    sinTheta *= -1.;
    rotateCoords(v);
    sinTheta *= -1.;
}

template class AARotate<utils::Axis::X>;
template class AARotate<utils::Axis::Y>;
template class AARotate<utils::Axis::Z>;