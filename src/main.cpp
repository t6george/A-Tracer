#include <iostream>
#include <memory>

#include <Utils.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>
#include <HittableList.hpp>
#include <Sphere.hpp>
#include <Camera.hpp>

const Vec3 WHITE(1., 1., 1.);
const Vec3 SKY_BLUE(.5, .7, 1.);
const Vec3 RED(1., 0., 0.);

void outputPPMGradient(const int width, const int height)
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

void outputSkyGradient(const int width, const int height, const int samplesPerPixel)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    Camera camera{Vec3{-2., -1., -1.}, Vec3{4., 0., 0.}, Vec3{0., 2., 0.}, Vec3{0., 0., 0.}};
    Hittable::HitRecord record;

    HittableList world;
    world.add(std::make_shared<Sphere>(Vec3{0., 0., -1.}, .5));
    world.add(std::make_shared<Sphere>(Vec3{0., -100.5, -1.}, 100.));
    Vec3 pixelColor;

    for (int i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int j = 0; j < width; ++j)
        {
            pixelColor.zero();
            for (int sample = 0; sample < samplesPerPixel; ++sample)
            {
                if (world.getCollisionData(camera.updateLineOfSight((j + random_double()) / width, (i + random_double()) / height), record, 0.))
                {
                    pixelColor += (record.normal + Vec3(1., 1., 1.)) / 2.;
                }
                else
                {
                    record.t = (camera.getLineOfSight().direction().getUnitVector().y() + 1.) / 2.;
                    pixelColor += (SKY_BLUE * record.t + WHITE * (1. - record.t));
                }
            }
            pixelColor.formatColor(std::cout, samplesPerPixel);
        }
    }
    std::cerr << std::endl;
}

int main()
{
    outputSkyGradient(200, 100, 100);
}