#pragma once

#include <Macro.cuh>
#include <Vec3.cuh>

class Pdf
{
public:
    DEV HOST Pdf() = default;
    DEV HOST virtual ~Pdf() noexcept = default;

    DEV virtual void construct(const Vec3& v) {}
    DEV virtual double eval(const Vec3& v) const = 0;
    DEV virtual Vec3 genRandomVector() const = 0;
};
