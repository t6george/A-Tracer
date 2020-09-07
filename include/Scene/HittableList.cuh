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

	HOST HittableNode(const SharedPointer<Hittable>& data = SharedPointer<Hittable>());
	HOST ~HittableNode() noexcept;
    };

    class HittableLinkedList
    {
        HittableNode* head;
        HittableNode* tail;
	unsigned len;
    
        public:
	HOST HittableLinkedList();
	HOST ~HittableLinkedList() noexcept;

	HOST void emplace_back(const SharedPointer<Hittable>& data);
	HOST void clear();
	HOST DEV bool empty() const;
	HOST DEV SharedPointer<Hittable> at(unsigned i) const;
	HOST DEV unsigned size() const;

	class Iterator;

	Iterator begin() const;
	Iterator end() const;

	class Iterator
	{
	    HittableNode* curr;
	public:
	    Iterator(HittableNode* n = nullptr);
	    ~Iterator() = default;

	    Iterator& operator++();
	    Iterator& operator=(HittableNode* n);
	    bool operator!=(const Iterator& it);
	    SharedPointer<Hittable> operator*();
	};
    };

    HittableLinkedList hittables;
#endif

public:
    HOST HittableList() = default;
    HOST ~HittableList() noexcept = default;

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
