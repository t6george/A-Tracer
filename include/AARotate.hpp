#pragma once
#include <Shape.hpp>

template <utils::Axis A>
class AARotate : public Hittable
{
    const std::shared_ptr<Shape> shape;
    const double sinTheta, cosTheta;
    AABB bbox;

    AABB computeBoundingBox(const std::shared_ptr<Shape> shape, double angle);
    void setCandidateExtreme(double x, double y, double z, Vec3 &extreme) const;

public:
    AARotate(const std::shared_ptr<Shape> shape, double angle);
    ~AARotate() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity) override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;
};