#include <CosinePdf.cuh>
#include <Util.cuh>
CosinePdf::CosinePdf() : basis{Vec3{}} {}

void CosinePdf::construct(const Vec3& v)
{
    basis = OrthonormalBasis{v};
}

double CosinePdf::eval(const Vec3& v) const
{
    return fmax(0., (v.getUnitVector()).o(basis.getW()) / utils::pi);
}

Vec3 CosinePdf::genRandomVector() const
{
    return basis.getVec(Vec3::randomCosineVec());
}