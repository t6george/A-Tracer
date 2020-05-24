#include <iostream>
#include <memory>

#include <Utils.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>
#include <HittableList.hpp>
#include <Sphere.hpp>
#include <Camera.hpp>
#include <LambertianDiffuse.hpp>
#include <Metal.hpp>
#include <Dielectric.hpp>
#include <vector>

const Vec3 WHITE(1., 1., 1.);
const Vec3 SKY_BLUE(.5, .7, 1.);
const Vec3 RED(1., 0., 0.);

Vec3 computeRayColor(const Ray &ray, HittableList &world, int depth)
{
    if (depth <= 0)
    {
        return Vec3{0., 0., 0.};
    }

    Hittable::HitRecord record;
    if (world.getCollisionData(ray, record, .001))
    {
        return computeRayColor(record.scatteredRay, world, depth - 1) * record.attenuation;
    }

    record.t = (ray.direction().getUnitVector().y() + 1.) / 2.;
    return (SKY_BLUE * record.t + WHITE * (1. - record.t));
}

void outputSphereScene(const int width, const int height, const int samplesPerPixel, const int maxReflections)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    Camera camera{static_cast<double>(width) / height, 20., .1, 10., Vec3{13., 2., 3.}, Vec3{0., 0., 0.}, 0., 1.};

    Hittable::HitRecord record;
    Vec3 pixelColor;
    HittableList world;
    Vec3 randomCenter0{0., .2, 0.};
    Vec3 randomCenter1;
    double chooseMaterial;

    world.add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000., std::make_shared<LambertianDiffuse>(Vec3{0., 127.5, 0.})));

    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            randomCenter0[0] = a + .9 * random_double();
            randomCenter0[2] = b + .9 * random_double();
            chooseMaterial = random_double();

            if ((randomCenter0 - Vec3{4., .2, 0.}).sqLen() > .81)
            {
                if (chooseMaterial < .7)
                {
                    randomCenter1 = randomCenter0;
                    randomCenter1[1] = random_double(0., .5);
                    world.add(std::make_shared<Sphere>(
                        randomCenter0, randomCenter1, .2, std::make_shared<LambertianDiffuse>(random_color()), 0., 1.));
                }
                else if (chooseMaterial < .9)
                {
                    world.add(std::make_shared<Sphere>(
                        randomCenter0, .2, std::make_shared<Metal>(random_color(Vec3{127.5, 127.5, 127.5}), random_double(0., .5))));
                }
                else
                {
                    world.add(std::make_shared<Sphere>(
                        randomCenter0, .2, std::make_shared<Dielectric>(1.5)));
                }
            }
        }
    }

    world.add(std::make_shared<Sphere>(Vec3{0., 1., 0.}, 1., std::make_shared<Dielectric>(1.5)));
    world.add(std::make_shared<Sphere>(Vec3{-4., 1., 0.}, 1., std::make_shared<LambertianDiffuse>(Vec3{102., 51., 25.5})));
    world.add(std::make_shared<Sphere>(Vec3{4., 1., 0.}, 1., std::make_shared<Metal>(Vec3{178.5, 153., 127.5}, 0.)));

    for (int i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int j = 0; j < width; ++j)
        {
            pixelColor.zero();
            for (int sample = 0; sample < samplesPerPixel; ++sample)
            {
                pixelColor += computeRayColor(camera.updateLineOfSight((j + random_double()) / width, (i + random_double()) / height),
                                              world, maxReflections);
            }
            pixelColor.formatColor(std::cout, samplesPerPixel);
        }
    }
    std::cerr << std::endl;
}

int main()
{
    outputSphereScene(384, 216, 100, 50);
}