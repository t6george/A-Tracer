#pragma once
#include <Hittable.hpp>
#include <AABB.hpp>

template <utils::Axis A>
class AARotate : public Hittable
{
    const std::shared_ptr<Hittable> shape;
    double sinTheta, cosTheta;
    AABB bbox;

    AABB computeBoundingBox();
    void rotateCoords(Vec3 &v, const double sin) const;
    void inverseRotateCoords(Vec3 &v, const double sin) const;

public:
    AARotate(const std::shared_ptr<Hittable> shape, double angle);
    ~AARotate() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record, WeightedPdf &pdf,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};