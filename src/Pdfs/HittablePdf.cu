#include <HittablePdf.cuh>

DEV HOST HittablePdf::HittablePdf(SharedPointer<Hittable> object)
 : object{object} {}


DEV void HittablePdf::construct(const Vec3& v)
{
    origin = v;
}

DEV double HittablePdf::eval(const Vec3& v) const
{
    return object->eval(origin, v);
}

DEV Vec3 HittablePdf::genRandomVector() const
{
    return object->genRandomVector(origin);
}