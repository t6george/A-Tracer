#pragma once
#include <vector>
#include <memory>

#include <Hittable.cuh>
#include <AABB.cuh>

class Ray;

class HittableList : public Hittable
{
    std::vector<std::shared_ptr<Hittable>> hittables;

public:
    DEV HOST HittableList() = default;
    DEV HOST ~HittableList() noexcept = default;

    DEV HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;

    DEV void add(std::shared_ptr<Hittable> hittable);
    DEV void clear();

    DEV Vec3 genRandomVector(const Vec3& origin) const override;
    DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;

    friend class BVHNode;
};