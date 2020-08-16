#pragma once

#include <Macro.hpp>
#include <Vec3.hpp>

class Pdf
{
public:
    virtual DEV HOST ~Pdf() noexcept = default;

    virtual DEV void construct(const Vec3& v) {}
    virtual DEV double eval(const Vec3& v) const = 0;
    virtual DEV Vec3 genRandomVector() const = 0;
};