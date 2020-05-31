#pragma once
#include <Hittable.hpp>

class AABB;

class FlipFace : public Hittable
{
    const std::shared_ptr<Hittable> hittable;

public:
    FlipFace(const std::shared_ptr<Hittable> hittable);
    ~FlipFace() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity,
                             double tMax = utils::infinity) override;
    bool getBoundingBox(double time0, double time1, AABB &box) const override;

    // virtual bool hit(const ray &r, double t_min, double t_max, Hittab &rec) const
    // {
    //     if (!ptr->hit(r, t_min, t_max, rec))
    //         return false;

    //     rec.front_face = !rec.front_face;
    //     return true;
    // }

    // virtual bool bounding_box(double t0, double t1, aabb &output_box) const
    // {
    //     return ptr->bounding_box(t0, t1, output_box);
    // }
};