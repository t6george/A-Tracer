#include <AARotate.hpp>

template <enum utils::Axis A>
AARotate<A>::AARotate(const std::shared_ptr<Shape> shape, double angle)
    : shape{shape}, sinTheta{sin(utils::deg_to_rad(angle))}, cosTheta{cos(utils::deg_to_rad(angle))},
      bbox{computeBoundingBox(shape, angle)} {}

template <enum utils::Axis A>
AABB AARotate<A>::computeBoundingBox(const std::shared_ptr<Shape> shape, double angle)
{
    AABB box;
    double x, y, z;
    shape->getBoundingBox(0., 1., box);

    Vec3 minPoint{-utils::infinity, -utils::infinity, -utils::infinity};
    Vec3 maxPoint{utils::infinity, utils::infinity, utils::infinity};
    Vec3 candidateExtreme;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                x = i * box.getMaxPoint().x() + (1 - i) * box.getMinPoint().x();
                y = j * box.getMaxPoint().y() + (1 - j) * box.getMinPoint().y();
                z = k * box.getMaxPoint().z() + (1 - k) * box.getMinPoint().z();

                setCandidateExtreme(x, y, z, candidateExtreme);

                for (int l = 0; l < 3; ++l)
                {
                    minPoint[l] = fmin(candidateExtreme[l], minPoint[l]);
                    maxPoint[l] = fmax(candidateExtreme[l], maxPoint[l]);
                }
            }
        }
    }
}

template <>
void AARotate<utils::Axis::X>::setCandidateExtreme(double x, double y, double z, Vec3 &extreme) const
{
    extreme = Vec3{
        x,
        cosTheta * y + sinTheta * z,
        cosTheta * z - sinTheta * y};
}

template <>
void AARotate<utils::Axis::Y>::setCandidateExtreme(double x, double y, double z, Vec3 &extreme) const
{
    extreme = Vec3{
        cosTheta * x + sinTheta * z,
        y,
        cosTheta * z - sinTheta * x};
}

template <>
void AARotate<utils::Axis::Z>::setCandidateExtreme(double x, double y, double z, Vec3 &extreme) const
{
    extreme = Vec3{
        cosTheta * x + sinTheta * y,
        cosTheta * y - sinTheta * x,
        z};
}

template class AARotate<utils::Axis::X>;
template class AARotate<utils::Axis::Y>;
template class AARotate<utils::Axis::Z>;
