#include <WeightedPdf.cuh>
#include <Util.cuh>
#include <cassert>

DEV HOST WeightedPdf::WeightedPdf(SharedPointer<Pdf> pdf1, SharedPointer<Pdf> pdf2, double pdf1Weight)
 : pdf1{pdf1}, pdf2{pdf2}, pdf1Weight{pdf1Weight} { assert(pdf1Weight >= 0 && pdf1Weight <= 1.); }

DEV SharedPointer<Pdf> WeightedPdf::getPdf1() const
{
    return pdf1;
}

DEV SharedPointer<Pdf> WeightedPdf::getPdf2() const
{
    return pdf2;
}

DEV double WeightedPdf::eval(const Vec3& v) const
{
    double prob = 0.;
    if (pdf1 && pdf2)
    {
        prob = pdf1Weight * pdf1->eval(v) + (1. - pdf1Weight) * pdf2->eval(v);
    }
    return prob;
}

DEV Vec3 WeightedPdf::genRandomVector() const
{
    Vec3 randomVec;
    if (pdf1 && pdf2)
    {
        randomVec = pdf1->genRandomVector();
        if (utils::random_double() > pdf1Weight)
        {
            randomVec = pdf2->genRandomVector();
        }
    }

    return randomVec;
}