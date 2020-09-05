#pragma once

#include <iostream>

#include <Macro.cuh>

class Vec3
{
    double c[3];

public:
    DEV HOST static Vec3 randomVector(const Vec3 &min = Vec3{0., 0., 0.}, const Vec3 &max = Vec3{1., 1., 1.});
    DEV HOST static Vec3 clamp(const Vec3 &v, const double min, const double max);
    DEV HOST static Vec3 randomUnitSphereVec();
    DEV HOST static Vec3 randomUnitHemisphereVec(const Vec3& normal);
    DEV HOST static Vec3 randomUnitCircleVec();
    DEV HOST static Vec3 randomCosineVec();
    DEV HOST static Vec3 randomVecToSphere(const double R, const double distSq);

    DEV HOST Vec3();
    DEV HOST Vec3(double c1, double c2, double c3);
    DEV HOST ~Vec3() noexcept = default;

    DEV HOST double x() const;
    DEV HOST double y() const;
    DEV HOST double z() const;

    DEV HOST double operator[](int i) const;
    DEV HOST double &operator[](int i);

    DEV HOST Vec3 &operator+=(const Vec3 &otherV);
    DEV HOST Vec3 &operator*=(double s);
    DEV HOST Vec3 &operator/=(double s);
    DEV HOST Vec3 operator-() const;

    DEV HOST double len() const;
    DEV HOST double sqLen() const;

    HOST void formatRaw(std::ostream &out) const;
    HOST void formatColor(std::ostream &out, int samplesPerPixel = 1);

    DEV HOST Vec3 operator+(const Vec3 &otherV) const;
    DEV HOST Vec3 operator-(const Vec3 &otherV) const;
    DEV HOST Vec3 operator*(const Vec3 &otherV) const;
    DEV HOST Vec3 operator*(double s) const;
    DEV HOST Vec3 operator/(double s) const;

    DEV HOST bool operator==(const Vec3 &otherV) const;
    DEV HOST bool operator!=(const Vec3 &otherV) const;

    DEV HOST Vec3 getUnitVector() const;

    DEV HOST double o(const Vec3 &otherV) const;
    DEV HOST Vec3 x(const Vec3 &otherV) const;
    DEV HOST Vec3 &operator*=(const Vec3 &otherV);
    DEV HOST void zero();

    DEV HOST Vec3 reflect(const Vec3 &normal) const;
    DEV HOST Vec3 refract(const Vec3 &normal, double n_over_nprime) const;
};

inline DEV Vec3 operator*(double s, const Vec3 &v) { return v * s; }
