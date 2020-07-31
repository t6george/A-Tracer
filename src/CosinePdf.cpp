#include <CosinePdf.hpp>
#include <Utils.hpp>

CosinePdf::CosinePdf(const Vec3& v) : basis{v} {}

double CosinePdf::eval(const Vec3& v) const
{
    return fmax(0., (v.getUnitVector()).o(basis.getW()) / utils::pi);
}

Vec3 CosinePdf::genRandomVector() const
{
    return basis.getVec(Vec3::randomCosineVec());
}