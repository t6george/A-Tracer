#pragma once
#include <Pdf.hpp>
#include <Vec3.hpp>
#include <Hittable.hpp>

class HittablePdf : public Pdf
{
    std::shared_ptr<Hittable> object;
    Vec3 origin;

public:
    HittablePdf(std::shared_ptr<Hittable> object, const Vec3& origin);
    ~HittablePdf() noexcept = default;

    double eval(const Vec3& v) const override;
    Vec3 genRandomVector() const override;
};