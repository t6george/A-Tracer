#pragma once
#include <Pdf.cuh>
#include <Vec3.cuh>
#include <Hittable.cuh>

class HittablePdf : public Pdf
{
    SharedPointer<Hittable> object;
    Vec3 origin;

public:
    HOST HittablePdf(SharedPointer<Hittable> object);
    HOST ~HittablePdf() noexcept = default;

    DEV virtual void construct(const Vec3& v) override;
    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};
