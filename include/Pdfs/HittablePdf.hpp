#pragma once
#include <Pdf.hpp>
#include <Vec3.hpp>
#include <Hittable.hpp>

class HittablePdf : public Pdf
{
    std::shared_ptr<Hittable> object;
    Vec3 origin;

public:
    DEV HOST HittablePdf(std::shared_ptr<Hittable> object);
    DEV HOST ~HittablePdf() noexcept = default;

    virtual DEV void construct(const Vec3& v) override;
    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};