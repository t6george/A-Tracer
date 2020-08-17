#pragma once

#include <array>

#include <Macro.cuh>

class Vec3
{
    std::array<double, 3> c;

public:
    static DEV Vec3 randomVector(const Vec3 &min = Vec3{0., 0., 0.}, const Vec3 &max = Vec3{1., 1., 1.});
    static DEV Vec3 clamp(const Vec3 &v, const double min, const double max);
    static DEV Vec3 randomUnitSphereVec();
    static DEV Vec3 randomUnitHemisphereVec(const Vec3& normal);
    static DEV Vec3 randomUnitCircleVec();
    static DEV Vec3 randomCosineVec();
    static DEV Vec3 randomVecToSphere(const double R, const double distSq);

    DEV HOST Vec3();
    DEV HOST Vec3(double c1, double c2, double c3);
    DEV HOST ~Vec3() noexcept = default;

    DEV double x() const;
    DEV double y() const;
    DEV double z() const;

    DEV double operator[](int i) const;
    DEV double &operator[](int i);

    DEV Vec3 &operator+=(const Vec3 &otherV);
    DEV Vec3 &operator*=(double s);
    DEV Vec3 &operator/=(double s);
    DEV Vec3 operator-() const;

    DEV double len() const;
    DEV double sqLen() const;

    DEV void formatRaw(std::ostream &out) const;
    DEV void formatColor(std::ostream &out, int samplesPerPixel = 1);

    DEV Vec3 operator+(const Vec3 &otherV) const;
    DEV Vec3 operator-(const Vec3 &otherV) const;
    DEV Vec3 operator*(const Vec3 &otherV) const;
    DEV Vec3 operator*(double s) const;
    DEV Vec3 operator/(double s) const;

    DEV bool operator==(const Vec3 &otherV) const;
    DEV bool operator!=(const Vec3 &otherV) const;

    DEV Vec3 getUnitVector() const;

    DEV double o(const Vec3 &otherV) const;
    DEV Vec3 x(const Vec3 &otherV) const;
    DEV Vec3 &operator*=(const Vec3 &otherV);
    DEV void zero();

    DEV Vec3 reflect(const Vec3 &normal) const;
    DEV Vec3 refract(const Vec3 &normal, double n_over_nprime) const;
};

inline DEV Vec3 operator*(double s, const Vec3 &v) { return v * s; }