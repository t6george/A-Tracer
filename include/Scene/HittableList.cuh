#pragma once

#if GPU == 0
#include <vector>
#endif

#include <Memory.cuh>
#include <Hittable.cuh>
#include <AABB.cuh>

class Ray;

class HittableList : public Hittable
{
#if GPU == 0
    std::vector<SharedPointer<Hittable>> hittables;
#else
    struct HittableNode;
    struct HittableNode
    {
        HittableNode* next;
        SharedPointer<Hittable> data;

        DEV HOST HittableNode(const SharedPointer<Hittable>& data = SharedPointer<Hittable>());
        DEV HOST ~HittableNode() noexcept;
    };

    class HittableLinkedList
    {
        HittableNode* head;
        HittableNode* tail;
	    unsigned len;
    
    public:
        DEV HOST HittableLinkedList();
        DEV HOST ~HittableLinkedList() noexcept;

        HOST void emplace_back(const SharedPointer<Hittable>& data);
        HOST void clear();
        DEV HOST bool empty() const;
        DEV HOST SharedPointer<Hittable> at(unsigned i) const;
        DEV HOST unsigned size() const;

        class Iterator;

        DEV HOST Iterator begin() const;
        DEV HOST Iterator end() const;

        class Iterator
        {
            HittableNode* curr;
        public:
            DEV HOST Iterator(HittableNode* n = nullptr);
            DEV HOST ~Iterator() = default;

            DEV HOST Iterator& operator++();
            DEV HOST Iterator& operator=(HittableNode* n);
            DEV HOST bool operator!=(const Iterator& it);
            DEV HOST SharedPointer<Hittable> operator*();
        };
    };

    HittableLinkedList hittables;
#endif

public:
    DEV HOST HittableList() = default;
    DEV HOST ~HittableList() noexcept = default;

    DEV Hittable::HitType getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin = -utils::infinity, double tMax = utils::infinity, 
                             bool flip = false) const override;

    DEV bool getBoundingBox(double time0, double time1, AABB &box) const override;

    HOST void add(SharedPointer<Hittable> hittable);
    HOST void clear();

    DEV Vec3 genRandomVector(const Vec3& origin) const override;
    DEV double eval(const Vec3& origin, const Vec3& v, bool flip = false) const override;

    friend class BVHNode;
};
