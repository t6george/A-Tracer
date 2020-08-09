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

    std::shared_ptr<Pdf> getPdf1() const;
    std::shared_ptr<Pdf> getPdf2() const;

    double eval(const Vec3& v) const override;
    Vec3 genRandomVector() const override;
};