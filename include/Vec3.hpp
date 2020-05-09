#pragma once
#include <array>

class Vec3
{
    std::array<double, 3> c;

public:
    Vec3();
    Vec3(double c1, double c2, double c3);
    ~Vec3() noexcept = default;

    double x() const;
    double y() const;
    double z() const;

    double operator[](int i) const;
    double &operator[](int i);

    Vec3 &operator+=(const Vec3 &otherV);
    Vec3 &operator*=(double s);
    Vec3 &operator/=(double s);
    Vec3 operator-() const;

    double len() const;
    double sqLen() const;

    void formatRaw(std::ostream &out) const;
    void formatColor(std::ostream &out, int samplesPerPixel = 1) const;

    Vec3 operator+(const Vec3 &otherV) const;
    Vec3 operator-(const Vec3 &otherV) const;
    Vec3 operator*(const Vec3 &otherV) const;
    Vec3 operator*(double s) const;
    Vec3 operator/(double s) const;

    Vec3 getUnitVector() const;

    double o(const Vec3 &otherV) const;
    Vec3 x(const Vec3 &otherV) const;
    void zero();

    Vec3 reflect(const Vec3 &normal) const;
};

inline Vec3 operator*(double s, const Vec3 &v) { return v * s; }