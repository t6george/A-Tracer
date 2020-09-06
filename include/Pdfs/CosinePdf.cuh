#pragma once
#include <Pdf.cuh>
#include <OrthonormalBasis.cuh>

class CosinePdf : public Pdf
{
    OrthonormalBasis basis;
public:
    HOST CosinePdf();
    HOST ~CosinePdf() noexcept = default;

    DEV void construct(const Vec3& v) override;
    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};
