#include <Shape.hpp>

Shape::Shape(const std::shared_ptr<Material> material,
             const double t0, const double t1)
    : material{material}, time0{t0}, time1{t1} {}