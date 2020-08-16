#pragma once
#include <Pdf.hpp>
#include <OrthonormalBasis.hpp>

class CosinePdf : public Pdf
{
    OrthonormalBasis basis;
public:
    DEV HOST CosinePdf();
    DEV HOST ~CosinePdf() noexcept = default;

    virtual DEV void construct(const Vec3& v) override;
    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};