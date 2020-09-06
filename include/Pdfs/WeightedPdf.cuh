#pragma once
#include <Pdf.cuh>
#include <Memory.cuh>

class WeightedPdf : public Pdf
{
    SharedPointer<Pdf> pdf1;
    SharedPointer<Pdf> pdf2;
    const double pdf1Weight;

public:
    DEV HOST WeightedPdf(SharedPointer<Pdf> pdf1, SharedPointer<Pdf> pdf2, double pdf1Weight);
    DEV HOST ~WeightedPdf() noexcept = default;

    DEV SharedPointer<Pdf> getPdf1() const;
    DEV SharedPointer<Pdf> getPdf2() const;

    DEV double eval(const Vec3& v) const override;
    DEV Vec3 genRandomVector() const override;
};