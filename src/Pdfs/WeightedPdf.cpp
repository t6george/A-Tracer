#include <WeightedPdf.hpp>
#include <Util.hpp>
#include <cassert>

WeightedPdf::WeightedPdf(std::shared_ptr<Pdf> pdf1, std::shared_ptr<Pdf> pdf2, double pdf1Weight)
 : pdf1{pdf1}, pdf2{pdf2}, pdf1Weight{pdf1Weight} { assert(pdf1Weight >= 0 && pdf1Weight <= 1.); }

double WeightedPdf::eval(const Vec3& v) const
{
    return pdf1Weight * pdf1->eval(v) + (1. - pdf1Weight) * pdf2->eval(v);
}

Vec3 WeightedPdf::genRandomVector() const
{
    Vec3 randomVec = pdf1->genRandomVector();
    if (utils::random_double() > pdf1Weight)
    {
        randomVec = pdf2->genRandomVector();
    }

    return randomVec;
}