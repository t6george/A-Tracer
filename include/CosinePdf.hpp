#pragma once
#include <Pdf.hpp>
#include <OrthonormalBasis.hpp>

class CosinePdf : public Pdf
{
    OrthonormalBasis basis;
public:
    CosinePdf(const Vec3& v);
    ~CosinePdf() noexcept = default;

    double eval(const Vec3& v) const override;
    Vec3 genRandomVector() const override;
};