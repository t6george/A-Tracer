#include <cassert>
#include <cstring>

#include <HittableList.cuh>
#include <AABB.cuh>

#if GPU == 1
DEV HOST HittableList::HittableNode::HittableNode(const SharedPointer<Hittable>& data) : next{nullptr}, data{data} {}

DEV HOST HittableList::HittableNode::~HittableNode()
{
    HittableNode* tmp = next;
    cudaFree(this);
    cudaFree(tmp);
}

DEV HOST HittableList::HittableLinkedList::HittableLinkedList() : head{nullptr}, tail{nullptr}, len{0} {}

DEV HOST HittableList::HittableLinkedList::~HittableLinkedList()
{
    clear();
}

DEV HOST void HittableList::HittableLinkedList::emplace_back(const SharedPointer<Hittable>& data)
{
    HittableNode* newNode = nullptr;
    HittableNode node(data);

    cudaMallocManaged((void**) &newNode, sizeof(HittableNode));
    memmove((void*) newNode, (void*) &node, sizeof(HittableNode));
    
    if (head)
    {
        tail->next = newNode;
	tail = tail->next;
    }
    else
    {
        head = tail = newNode;
    }

    ++len;
}

DEV HOST void HittableList::HittableLinkedList::clear()
{
    cudaFree(head);
    head = tail = nullptr;
    len = 0;
}

DEV HOST bool HittableList::HittableLinkedList::empty() const
{
    return len == 0;
}

DEV HOST SharedPointer<Hittable> HittableList::HittableLinkedList::at(unsigned i) const
{
    SharedPointer<Hittable> hittable;
    HittableNode* itr = head;

    for (unsigned j = 0; j < i && itr; ++j)
    {
        itr = itr->next;
    }
    
    if (itr)
    {
        hittable = itr->data;
    }

    return hittable;
}

DEV HOST HittableList::HittableLinkedList::Iterator HittableList::HittableLinkedList::begin() const
{
    return HittableLinkedList::Iterator(head);
}

DEV HOST HittableList::HittableLinkedList::Iterator HittableList::HittableLinkedList::end() const
{
    return HittableLinkedList::Iterator(nullptr);
}

DEV HOST HittableList::HittableLinkedList::Iterator::Iterator(HittableNode* n) : curr{n} {}

DEV HOST HittableList::HittableLinkedList::Iterator& HittableList::HittableLinkedList::Iterator::operator++()
{
    if (curr)
    {
        curr = curr->next;
    }

    return *this;
}

DEV HOST HittableList::HittableLinkedList::Iterator& HittableList::HittableLinkedList::Iterator::operator=(HittableNode* n)
{
    curr = n;
    return *this;
}

DEV HOST bool HittableList::HittableLinkedList::Iterator::operator!=(const Iterator& it)
{
    return it.curr != curr;
}

DEV HOST SharedPointer<Hittable> HittableList::HittableLinkedList::Iterator::operator*()
{
    return curr->data;
}

#endif

DEV Hittable::HitType HittableList::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    Hittable::HitRecord tmpRecord;
    Hittable::HitType collisionType = Hittable::HitType::NO_HIT, tmpCollisionType;

    for (const auto &obj : hittables)
    {
        if (static_cast<bool>(tmpCollisionType = obj.get()->getCollisionData(ray, tmpRecord, tMin, tMax, flip)))
        {
            collisionType = tmpCollisionType;
            record = tmpRecord;
            tMax = record.t;
        }
    }

    return collisionType;
}

DEV bool HittableList::getBoundingBox(double time0, double time1, AABB &box) const
{
    bool firstBox = true;
    AABB tmp, outputBox;

    for (const auto &obj : hittables)
    {
        if (!obj->getBoundingBox(time0, time1, tmp))
            return false;
        outputBox = firstBox ? tmp : AABB::combineAABBs(outputBox, tmp);
        firstBox = false;
    }

    return !hittables.empty();
}

HOST void HittableList::add(SharedPointer<Hittable> hittable)
{
    hittables.emplace_back(hittable);
}

HOST void HittableList::clear()
{
    hittables.clear(); 
}

DEV Vec3 HittableList::genRandomVector(const Vec3& origin) const
{
    return hittables.at(utils::random_int(0, hittables.size()))->genRandomVector(origin);
}

DEV double HittableList::eval(const Vec3& origin, const Vec3& v, bool flip) const
{
    assert(hittables.size() > 0);
    double weight = 1. / hittables.size();
    double sum = 0.;

    for (const auto& hittable : hittables)
    {
        sum += weight * hittable->eval(origin, v, flip);
    }

    return sum;
}
