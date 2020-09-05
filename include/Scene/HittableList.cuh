#pragma once
#include <vector>
#include <SharedPointer.cuh>

#include <Hittable.cuh>
#include <AABB.cuh>

class Ray;

class HittableList : public Hittable
{
    std::vector<SharedPointer<Hittable>> hittables;

public:
    HOST HittableList() = default;
    HOST ~HittableList() noexcept = default;

    DEV Hittable::HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;

    DEV void add(SharedPointer<Hittable> hittable);
    DEV void clear();

    DEV Vec3 genRandomVector(const Vec3& origin) const override;
    DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;

    friend class BVHNode;
};
