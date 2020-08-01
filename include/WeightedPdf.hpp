#pragma once
#include <Pdf.hpp>
#include <memory>

class WeightedPdf : public Pdf
{
    std::shared_ptr<Pdf> pdf1;
    std::shared_ptr<Pdf> pdf2;
    const double pdf1Weight;

public:
    WeightedPdf(std::shared_ptr<Pdf> pdf1, std::shared_ptr<Pdf> pdf2, double pdf1Weight);
    ~WeightedPdf() noexcept = default;

    double eval(const Vec3& v) const override;
    Vec3 genRandomVector() const override;
};