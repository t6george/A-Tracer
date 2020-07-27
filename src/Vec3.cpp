#include <Vec3.hpp>

#include <iostream>
#include <Utils.hpp>

Vec3::Vec3() : c{0.0, 0.0, 0.0} {}

Vec3::Vec3(double c1, double c2, double c3) : c{c1, c2, c3} {}

double Vec3::x() const { return c[0]; }

double Vec3::y() const { return c[1]; }

double Vec3::z() const { return c[2]; }

double Vec3::operator[](int i) const { return c.at(i); }

double &Vec3::operator[](int i) { return c.at(i); }

Vec3 &Vec3::operator+=(const Vec3 &otherV)
{
    c[0] += otherV.x();
    c[1] += otherV.y();
    c[2] += otherV.z();
    return *this;
}

Vec3 &Vec3::operator*=(double s)
{
    c[0] *= s;
    c[1] *= s;
    c[2] *= s;
    return *this;
}

Vec3 &Vec3::operator/=(double s)
{
    c[0] /= s;
    c[1] /= s;
    c[2] /= s;
    return *this;
}

Vec3 Vec3::operator-() const { return Vec3(-c[0], -c[1], -c[2]); }

double Vec3::len() const { return sqrt(sqLen()); }

double Vec3::sqLen() const { return c[0] * c[0] + c[1] * c[1] + c[2] * c[2]; }

void Vec3::formatRaw(std::ostream &out) const
{
    out << c[0] << ' '
        << c[1] << ' '
        << c[2] << '\n';
}

void Vec3::formatColor(std::ostream &out, int samplesPerPixel) const
{
    // auto r = utils::clamp(sqrt(c[0] / samplesPerPixel), 0., .999);
    out << static_cast<int>(256 * utils::clamp(sqrt(c[0] / samplesPerPixel), 0., .999)) << ' '
        << static_cast<int>(256 * utils::clamp(sqrt(c[1] / samplesPerPixel), 0., .999)) << ' '
        << static_cast<int>(256 * utils::clamp(sqrt(c[2] / samplesPerPixel), 0., .999)) << '\n';
    // if (r > .01 && r < .9)
    // {
    //     auto g = utils::clamp(sqrt(c[1] / samplesPerPixel), 0., .999);
    //     auto b = utils::clamp(sqrt(c[2] / samplesPerPixel), 0., .999);
    //     std::cerr << static_cast<int>(256. * r) << ' '
    //               << static_cast<int>(256. * g) << ' '
    //               << static_cast<int>(256. * b) << '\n';
    // }
}

Vec3 Vec3::operator+(const Vec3 &otherV) const
{
    return Vec3{c[0] + otherV.x(), c[1] + otherV.y(), c[2] + otherV.z()};
}

Vec3 Vec3::operator-(const Vec3 &otherV) const
{
    return Vec3{c[0] - otherV.x(), c[1] - otherV.y(), c[2] - otherV.z()};
}

Vec3 Vec3::operator*(const Vec3 &otherV) const
{
    return Vec3{c[0] * otherV.x(), c[1] * otherV.y(), c[2] * otherV.z()};
}

Vec3 Vec3::operator*(double s) const
{
    return Vec3{c[0] * s, c[1] * s, c[2] * s};
}

Vec3 Vec3::operator/(double s) const
{
    return Vec3{c[0] / s, c[1] / s, c[2] / s};
}

bool Vec3::operator==(const Vec3 &otherV) const
{
    return c[0] == otherV.x() && c[1] == otherV.y() && c[2] == otherV.z();
}

bool Vec3::operator!=(const Vec3 &otherV) const { return !(*this == otherV); }

Vec3 Vec3::getUnitVector() const { return *this / len(); }

double Vec3::o(const Vec3 &otherV) const
{
    return c[0] * otherV.x() + c[1] * otherV.y() + c[2] * otherV.z();
}

Vec3 Vec3::x(const Vec3 &otherV) const
{
    return Vec3(c[1] * otherV.z() - c[2] * otherV.y(),
                c[2] * otherV.x() - c[0] * otherV.z(),
                c[0] * otherV.y() - c[1] * otherV.x());
}

void Vec3::zero() { c[0] = c[1] = c[2] = 0.; }

Vec3 Vec3::reflect(const Vec3 &normal) const { return *this - 2 * o(normal) * normal; }

Vec3 Vec3::refract(const Vec3 &normal, double n_over_nprime) const
{
    Vec3 parallel = n_over_nprime * (*this + normal * normal.o(-*this));
    return parallel - normal * sqrt(1. - parallel.sqLen());
}

Vec3 Vec3::randomVector(const Vec3 &min, const Vec3 &max)
{
    return Vec3{utils::random_double(min[0], max[0]), utils::random_double(min[1], max[1]), utils::random_double(min[2], max[2])};
}

Vec3 Vec3::clamp(const Vec3 &v, const double min, const double max)
{
    return Vec3{utils::clamp(v[0], min, max), utils::clamp(v[1], min, max), utils::clamp(v[2], min, max)};
}

Vec3 Vec3::randomUnitSphereVec()
{
    double a = utils::random_double(0, 2 * utils::pi);
    double z = utils::random_double(-1., 1.);
    double r = sqrt(1. - z * z);
    return Vec3{r * cos(a), r * sin(a), z};
}

Vec3 Vec3::randomUnitHemisphereVec(const Vec3& normal)
{
    Vec3 v = Vec3::randomUnitSphereVec();
    if (v.o(normal) < 0.)
    {
        v = -v;
    }

    return v;
}

Vec3 Vec3::randomUnitCircleVec()
{
    Vec3 vec;
    vec[0] = utils::random_double();
    vec[1] = utils::random_double(0., sqrt(1. - vec[0] * vec[0]));
    return vec;
}