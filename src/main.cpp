#include <iostream>
#include <cstdint>

#include <Vec3.hpp>
#include <Ray.hpp>
#include <Surface.hpp>
#include <Sphere.hpp>

const Vec3 WHITE(1., 1., 1.);
const Vec3 SKY_BLUE(.5, .7, 1.);
const Vec3 RED(1., 0., 0.);

void outputPPMGradient(const uint16_t width, const uint16_t height)
{
    Vec3 colorV;
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";
    for (int32_t i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int32_t j = 0; j < width; ++j)
        {
            colorV[0] = static_cast<double>(j) / width;
            colorV[1] = static_cast<double>(i) / height;
            colorV[2] = .2;

            colorV.formatColor(std::cout);
        }
    }
    std::cerr << std::endl;
}

void outputSkyGradient(const uint16_t width, const uint16_t height)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";
    Vec3 origin{-2., -1., -1.};
    Vec3 planeWidth{4., 0., 0.};
    Vec3 planeHeight{0., 2., 0.};
    Vec3 camera{0., 0., 0.};
    Ray ray{camera};
    Sphere sphere(Vec3{0., 0., -1.}, .5);
    Surface::HitRecord record;

    for (int32_t i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int32_t j = 0; j < width; ++j)
        {
            ray.resetDirection(origin + static_cast<double>(j) / width * planeWidth + static_cast<double>(i) / height * planeHeight);
            if (sphere.getCollisionData(ray, record))
            {
                ((record.normal + Vec3(1., 1., 1.)) / 2.).formatColor(std::cout);
            }
            else
            {
                record.t = (ray.direction().getUnitVector().y() + 1.) / 2.;
                (SKY_BLUE * record.t + WHITE * (1. - record.t)).formatColor(std::cout);
            }
        }
    }
    std::cerr << std::endl;
}

int main()
{
    outputSkyGradient(200, 100);
}