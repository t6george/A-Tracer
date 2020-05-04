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

// absorbtion should be a material property
Vec3 computeRayColor(const Ray &ray, const HittableList &world, int depth, double absorbtion)
{
    if (depth <= 0)
    {
        return Vec3{0., 0., 0.};
    }

    Hittable::HitRecord record;
    if (world.getCollisionData(ray, record, .001))
    {
        return computeRayColor(Ray{record.point, record.normal + random_unit_vec()},
                               world, depth - 1, absorbtion) *
               absorbtion;
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
                pixelColor += computeRayColor(camera.updateLineOfSight((j + random_double()) / width, (i + random_double()) / height),
                                              world, maxReflections, .5);
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