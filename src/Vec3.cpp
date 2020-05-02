#include <Vec3.hpp>

#include <math.h>
#include <iostream>

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

void Vec3::formatColor(std::ostream &out) const
{
    out << static_cast<int>(255.999 * c[0]) << ' '
        << static_cast<int>(255.999 * c[1]) << ' '
        << static_cast<int>(255.999 * c[2]) << '\n';
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