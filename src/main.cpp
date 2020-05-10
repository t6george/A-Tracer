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

const Vec3 WHITE(1., 1., 1.);
const Vec3 SKY_BLUE(.5, .7, 1.);
const Vec3 RED(1., 0., 0.);

Vec3 computeRayColor(const Ray &ray, const HittableList &world, int depth)
{
    if (depth <= 0)
    {
        return Vec3{0., 0., 0.};
    }

    Hittable::HitRecord record;
    if (world.getCollisionData(ray, record, .001))
    {
        return computeRayColor(record.reflectedRay, world, depth - 1) * record.attenuation;
    }

    record.t = (ray.direction().getUnitVector().y() + 1.) / 2.;
    return (SKY_BLUE * record.t + WHITE * (1. - record.t));
}

void outputSphereScene(const int width, const int height, const int samplesPerPixel, const int maxReflections)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    Camera camera{static_cast<double>(width) / height, 90.};
    Hittable::HitRecord record;
    Vec3 pixelColor;
    HittableList world;

    double R = cos(pi / 4);
    world.add(std::make_shared<Sphere>(Vec3{-R, 0., -1.}, R, LambertianDiffuse{Vec3{0., 0., 255.}}));
    world.add(std::make_shared<Sphere>(Vec3(R, 0., -1.), R, LambertianDiffuse{Vec3{255., 0., 0.}}));

    // world.add(std::make_shared<Sphere>(Vec3{0., 0., -1.}, .5, LambertianDiffuse{Vec3{179.2, 76.8, 76.8}}));
    // world.add(std::make_shared<Sphere>(Vec3{0., -100.5, -1.}, 100., LambertianDiffuse{Vec3{204.8, 204.8, 0.}}));
    // world.add(std::make_shared<Sphere>(Vec3{1, 0, -1}, .5, Metal{Vec3{204.8, 153.6, 51.2}, .3}));
    // world.add(std::make_shared<Sphere>(Vec3{-1, 0, -1}, .5, Metal{Vec3{204.8, 204.8, 204.8}, .7}));
    // world.add(std::make_shared<Sphere>(Vec3{-1, 0, -1}, .5, Dielectric{1.5}));

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