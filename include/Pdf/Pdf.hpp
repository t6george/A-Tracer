#pragma once
#include <Vec3.hpp>

class Pdf
{
public:
    virtual ~Pdf() noexcept = default;

    virtual double eval(const Vec3& v) const = 0;
    virtual Vec3 genRandomVector() const = 0;
};