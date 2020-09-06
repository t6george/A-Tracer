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
    typedef struct _HittableNode
    {
	_HittableNode* next;
        SharedPointer<Hittable> data;

	HOST HittableNode(const SharedPointer<Hittable>& data = SharedPointer<Hittable>());
	HOST ~HittableNode() noexcept;
    } HittableNode;

    class HittableLinkedList
    {
        HittableNode* head;
        HittableNode* tail;
	unsigned len;
    
        public:
	HittableLinkedList();
	~HittableLinkedList() noexcept;

	void emplace_back(const SharedPointer<Hittable>& data);
	void clear();
	bool empty() const;
	SharedPointer<Hittable> at(unsigned i) const;
	unsigned size() const;
    };

    HittableList hittables;
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
