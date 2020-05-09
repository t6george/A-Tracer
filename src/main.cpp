#include <iostream>
#include <memory>

#include <Utils.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>
#include <HittableList.hpp>
#include <Sphere.hpp>
#include <Camera.hpp>
#include <LambertianDiffuse.hpp>

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
        // ray.resetOrigin(record.point);
        // ray.resetDirection(record.normal + random_unit_vec());
        return computeRayColor(ray, world, depth - 1) * .5;
    }

    record.t = (ray.direction().getUnitVector().y() + 1.) / 2.;
    return (SKY_BLUE * record.t + WHITE * (1. - record.t));
}

void outputSphereScene(const int width, const int height, const int samplesPerPixel, const int maxReflections)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    Camera camera{Vec3{-2., -1., -1.}, Vec3{4., 0., 0.}, Vec3{0., 2., 0.}, Vec3{0., 0., 0.}};
    Hittable::HitRecord record;

    HittableList world;
    const Material &diffMat = LambertianDiffuse{Vec3{255., 0., 0.}};
    world.add(std::make_shared<Sphere>(Vec3{0., 0., -1.}, .5, diffMat));
    world.add(std::make_shared<Sphere>(Vec3{0., -100.5, -1.}, 100., diffMat));
    Vec3 pixelColor;

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
    outputSphereScene(200, 100, 100, 50);
}