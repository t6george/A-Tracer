#pragma once
#include <array>

class Vec3
{
    std::array<double, 3> c;

public:
    static Vec3 randomVector(const Vec3 &min = Vec3{0., 0., 0.}, const Vec3 &max = Vec3{1., 1., 1.});
    static Vec3 clamp(const Vec3 &v, const double min, const double max);
    static Vec3 randomUnitSphereVec();
    static Vec3 randomUnitHemisphereVec(const Vec3& normal);
    static Vec3 randomUnitCircleVec();
    static Vec3 randomCosineVec();
    static Vec3 randomVecToSphere(const double R, const double distSq);

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
    void formatColor(std::ostream &out, int samplesPerPixel = 1);

    Vec3 operator+(const Vec3 &otherV) const;
    Vec3 operator-(const Vec3 &otherV) const;
    Vec3 operator*(const Vec3 &otherV) const;
    Vec3 operator*(double s) const;
    Vec3 operator/(double s) const;

    bool operator==(const Vec3 &otherV) const;
    bool operator!=(const Vec3 &otherV) const;

    Vec3 getUnitVector() const;

    double o(const Vec3 &otherV) const;
    Vec3 x(const Vec3 &otherV) const;
    void zero();

    Vec3 reflect(const Vec3 &normal) const;
    Vec3 refract(const Vec3 &normal, double n_over_nprime) const;
};

inline Vec3 operator*(double s, const Vec3 &v) { return v * s; }