#include <HittablePdf.hpp>

HittablePdf::HittablePdf(std::shared_ptr<Hittable> object, const Vec3& origin)
 : object{object}, origin{origin} {}


double HittablePdf::eval(const Vec3& v) const
{
    return object->eval(origin, v);
}

Vec3 HittablePdf::genRandomVector() const
{
    return object->genRandomVector(origin);
}