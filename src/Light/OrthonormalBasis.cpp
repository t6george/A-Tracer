#include <OrthonormalBasis.cuh>
#include <cmath>

Vec3 OrthonormalBasis::operator[](int i) const { return axes.at(i); } 

OrthonormalBasis::OrthonormalBasis(const Vec3& sample) : axes{Vec3{}, Vec3{}, sample.getUnitVector()}
{
    Vec3 tmp = fabs(getW().x()) > .9 ? Vec3{0., 1., 0.} : Vec3{1., 0., 0.};
    axes[1] = (getW().x(tmp)).getUnitVector();
    axes[0] = getW().x(getV());
}

Vec3 OrthonormalBasis::getU() const { return axes[0]; }
Vec3 OrthonormalBasis::getV() const { return axes[1]; }
Vec3 OrthonormalBasis::getW() const { return axes[2]; }

Vec3 OrthonormalBasis::getVec(Vec3 v) const { return getVec(v.x(), v.y(), v.z()); }

Vec3 OrthonormalBasis::getVec(double a, double b, double c) const
{
    return a * getU() + b * getV() + c * getW();
}

