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
    HittableList() = default;
    ~HittableList() noexcept = default;

    HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    bool getBoundingBox(double time0, double time1, AABB &box) const override;

    void add(std::shared_ptr<Hittable> hittable);
    void clear();

    Vec3 genRandomVector(const Vec3& origin) const override;
    double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;

    friend class BVHNode;
};