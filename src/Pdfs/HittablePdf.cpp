#include <HittablePdf.cuh>

HittablePdf::HittablePdf(std::shared_ptr<Hittable> object)
 : object{object} {}


void HittablePdf::construct(const Vec3& v)
{
    origin = v;
}

double HittablePdf::eval(const Vec3& v) const
{
    return object->eval(origin, v);
}

Vec3 HittablePdf::genRandomVector() const
{
    return object->genRandomVector(origin);
}