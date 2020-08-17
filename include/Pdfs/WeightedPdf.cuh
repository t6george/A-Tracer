#pragma once
#include <Pdf.cuh>
#include <memory>

class WeightedPdf : public Pdf
{
    std::shared_ptr<Pdf> pdf1;
    std::shared_ptr<Pdf> pdf2;
    const double pdf1Weight;

public:
    DEV HOST WeightedPdf(std::shared_ptr<Pdf> pdf1, std::shared_ptr<Pdf> pdf2, double pdf1Weight);
    DEV HOST ~WeightedPdf() noexcept = default;

    DEV std::shared_ptr<Pdf> getPdf1() const;
    DEV std::shared_ptr<Pdf> getPdf2() const;

    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};