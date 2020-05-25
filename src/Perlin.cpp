#include <Perlin.hpp>
#include <Utils.hpp>

template <typename T, size_t N>
void Perlin::permuteArray(std::array<T, N> &arr)
{
    int tgt;
    for (size_t i = N - 1; i > 0; --i)
    {
        tgt = random_int(0, i + 1);
        std::swap(arr[i], arr[tgt]);
    }
}

Perlin::Perlin() { init(); }

void Perlin::init()
{
    for (int i = 0; i < Perlin::pointCount; ++i)
    {
        randomDoubles[i] = random_double();
        permX[i] = permY[i] = permZ[i] = i;
    }

    Perlin::permuteArray(permX);
    Perlin::permuteArray(permY);
    Perlin::permuteArray(permZ);
}

double Perlin::getNoise(const Vec3 &point) const
{
    int i = static_cast<int>(4 * point.x()) & 0xFF;
    int j = static_cast<int>(4 * point.y()) & 0xFF;
    int k = static_cast<int>(4 * point.z()) & 0xFF;

    return randomDoubles[permX[i] ^ permY[j] ^ permZ[k]];
}