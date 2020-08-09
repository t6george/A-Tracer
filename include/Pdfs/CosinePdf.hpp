#pragma once
#include <Pdf.hpp>
#include <OrthonormalBasis.hpp>

class CosinePdf : public Pdf
{
    OrthonormalBasis basis;
public:
    CosinePdf();
    ~CosinePdf() noexcept = default;

    virtual void construct(const Vec3& v) override;
    double eval(const Vec3& v) const override;
    Vec3 genRandomVector() const override;
};