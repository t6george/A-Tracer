#include <CosinePdf.cuh>
#include <Util.cuh>

DEV HOST CosinePdf::CosinePdf() : basis{Vec3{}} {}

DEV void CosinePdf::construct(const Vec3& v)
{
    basis = OrthonormalBasis{v};
}

DEV double CosinePdf::eval(const Vec3& v) const
{
    return fmax(0., (v.getUnitVector()).o(basis.getW()) / utils::pi);
}

DEV Vec3 CosinePdf::genRandomVector() const
{
    return basis.getVec(Vec3::randomCosineVec());
}