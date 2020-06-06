#include <AARect.hpp>
#include <Material.hpp>
#include <AABB.hpp>

template <enum utils::Axis A>
AARect<A>::AARect(const double i0, const double i1, const double j0,
                  const double j1, const double k,
                  const std::shared_ptr<Material> material)
    : Shape::Shape{material, AABB{Vec3{i0, j0, k - .0001}, Vec3{i1, j1, k + .0001}}},
      i0{i0}, i1{i1}, j0{j0}, j1{j1}, k{k} {}

template <enum utils::Axis A>
Hittable::HitType AARect<A>::getCollisionData(const Ray &ray, Hittable::HitRecord &record, double tMin, double tMax)
{
    double t;
    solveForTime(ray, t);
    Hittable::HitType hit = Hittable::HitType::NO_HIT;

    if (t <= tMax && t >= tMin)
    {
        double i, j;
        getPlaneIntersection(ray, i, j, t);
        if (i >= i0 && i <= i1 && j >= j0 && j <= j1)
        {
            record.t = t;
            record.u = (i - i0) / (i1 - i0);
            record.v = (j - j0) / (j1 - j0);
            setHitPoint(i, j, k, record);
            record.setLightPosition(ray);

            hit = material->scatterRay(ray, record) ? Hittable::HitType::HIT_SCATTER
                                                    : Hittable::HitType::HIT_NO_SCATTER;
        }
    }

    return hit;
}

template <>
void AARect<utils::Axis::X>::solveForTime(const Ray &ray, double &t) const
{
    t = (k - ray.getOrigin().x()) / ray.getDirection().x();
}

template <>
void AARect<utils::Axis::Y>::solveForTime(const Ray &ray, double &t) const
{
    t = (k - ray.getOrigin().y()) / ray.getDirection().y();
}

template <>
void AARect<utils::Axis::Z>::solveForTime(const Ray &ray, double &t) const
{
    t = (k - ray.getOrigin().z()) / ray.getDirection().z();
}

template <>
void AARect<utils::Axis::X>::getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const
{
    i = ray.getOrigin().y() + ray.getDirection().y() * t;
    j = ray.getOrigin().z() + ray.getDirection().z() * t;
}

template <>
void AARect<utils::Axis::Y>::getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const
{
    i = ray.getOrigin().x() + ray.getDirection().x() * t;
    j = ray.getOrigin().z() + ray.getDirection().z() * t;
}

template <>
void AARect<utils::Axis::Z>::getPlaneIntersection(const Ray &ray, double &i, double &j, const double t) const
{
    i = ray.getOrigin().x() + ray.getDirection().x() * t;
    j = ray.getOrigin().y() + ray.getDirection().y() * t;
}

template <>
void AARect<utils::Axis::X>::setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const
{
    record.normal = Vec3{1., 0., 0.};
    record.point = Vec3{k, i, j};
}

template <>
void AARect<utils::Axis::Y>::setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const
{
    record.normal = Vec3{0., 1., 0.};
    record.point = Vec3{i, k, j};
}

template <>
void AARect<utils::Axis::Z>::setHitPoint(const double i, const double j, const double k, Hittable::HitRecord &record) const
{
    record.normal = Vec3{0., 0., 1.};
    record.point = Vec3{i, j, k};
}

template class AARect<utils::Axis::X>;
template class AARect<utils::Axis::Y>;
template class AARect<utils::Axis::Z>;