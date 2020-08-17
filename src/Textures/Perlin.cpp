#include <Perlin.cuh>
#include <Util.cuh>

double Perlin::trilinearInterpolation(const double c[2][2][2],
                                      double u,
                                      double v,
                                      double w) const
{
    double a = 0., di, dj, dk;
    u = u * u * (3. - 2. * u);
    v = v * v * (3. - 2. * v);
    w = w * w * (3. - 2. * w);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                di = static_cast<double>(i);
                dj = static_cast<double>(j);
                dk = static_cast<double>(k);

                a += c[i][j][k] *
                     (di * u + (1. - di) * (1. - u)) *
                     (dj * v + (1. - dj) * (1. - v)) *
                     (dk * w + (1. - dk) * (1. - w));
            }
        }
    }

    return a;
}

double Perlin::perlinInterpolation(const Vec3 c[2][2][2],
                                   double u,
                                   double v,
                                   double w) const
{
    double a = 0., di, dj, dk;
    u = u * u * (3. - 2. * u);
    v = v * v * (3. - 2. * v);
    w = w * w * (3. - 2. * w);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                di = static_cast<double>(i);
                dj = static_cast<double>(j);
                dk = static_cast<double>(k);

                a += c[i][j][k].o(Vec3{u - di, v - dj, w - dk}) *
                     (di * u + (1. - di) * (1. - u)) *
                     (dj * v + (1. - dj) * (1. - v)) *
                     (dk * w + (1. - dk) * (1. - w));
            }
        }
    }

    return a;
}

template <typename T, size_t N>
void Perlin::permuteArray(std::array<T, N> &arr)
{
    int tgt;
    for (size_t i = N - 1; i > 0; --i)
    {
        tgt = utils::random_int(0, i + 1);
        std::swap(arr[i], arr[tgt]);
    }
}

Perlin::Perlin() { init(); }

void Perlin::init()
{
    for (int i = 0; i < Perlin::pointCount; ++i)
    {
        randomVectors[i] = Vec3{utils::random_double(-1., 1.), utils::random_double(-1., 1.), utils::random_double(-1., 1.)}.getUnitVector();
        // randomDoubles[i] = random_double();
        permX[i] = permY[i] = permZ[i] = i;
    }

    Perlin::permuteArray(permX);
    Perlin::permuteArray(permY);
    Perlin::permuteArray(permZ);
}

double Perlin::getScalarNoise(const Vec3 &point) const
{
    int i = static_cast<int>(floor(point.x()));
    int j = static_cast<int>(floor(point.y()));
    int k = static_cast<int>(floor(point.z()));

    double u = point.x() - static_cast<double>(i);
    double v = point.y() - static_cast<double>(j);
    double w = point.z() - static_cast<double>(k);

    double c[2][2][2];

    for (int di = 0; di < 2; ++di)
    {
        for (int dj = 0; dj < 2; ++dj)
        {
            for (int dk = 0; dk < 2; ++dk)
            {
                c[di][dj][dk] = randomDoubles[permX[(i + di) & 0xFF] ^
                                              permY[(j + dj) & 0xFF] ^
                                              permZ[(k + dk) & 0xFF]];
            }
        }
    }

    return trilinearInterpolation(c, u, v, w);
}

double Perlin::getLaticeVectorNoise(const Vec3 &point) const
{
    int i = static_cast<int>(floor(point.x()));
    int j = static_cast<int>(floor(point.y()));
    int k = static_cast<int>(floor(point.z()));

    double u = point.x() - static_cast<double>(i);
    double v = point.y() - static_cast<double>(j);
    double w = point.z() - static_cast<double>(k);

    Vec3 c[2][2][2];

    for (int di = 0; di < 2; ++di)
    {
        for (int dj = 0; dj < 2; ++dj)
        {
            for (int dk = 0; dk < 2; ++dk)
            {
                c[di][dj][dk] = randomVectors[permX[(i + di) & 0xFF] ^
                                              permY[(j + dj) & 0xFF] ^
                                              permZ[(k + dk) & 0xFF]];
            }
        }
    }

    return perlinInterpolation(c, u, v, w);
}

double Perlin::getTurbulence(const Vec3 &point, int depth) const
{
    double a = 0., weight = 1.;
    Vec3 tmp = point;

    for (int i = 0; i < depth; ++i)
    {
        a += weight * getLaticeVectorNoise(point);
        weight /= 2.;
        tmp *= 2.;
    }

    return fabs(a);
}