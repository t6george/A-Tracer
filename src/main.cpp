#include <iostream>
#include <memory>

#include <Utils.hpp>
#include <SolidColor.hpp>
#include <CheckerTexture.hpp>
#include <TurbulentTexture.hpp>
#include <ImageTexture.hpp>
#include <DiffuseLight.hpp>
#include <AARect.hpp>
#include <PerlinNoiseTexture.hpp>
#include <FlipFace.hpp>
#include <Box.hpp>
#include <AARotate.hpp>
#include <Translate.hpp>

#include <Ray.hpp>
#include <HittableList.hpp>
#include <Sphere.hpp>
#include <Camera.hpp>
#include <LambertianDiffuse.hpp>
#include <Metal.hpp>
#include <Dielectric.hpp>
#include <vector>

const Vec3 WHITE{1., 1., 1.};
const Vec3 SKY_BLUE{.5, .7, 1.};
const Vec3 RED{1., 0., 0.};

Vec3 computeRayColor(const Ray &ray, const Vec3 &background, HittableList &world, int depth)
{
    Vec3 color;
    if (depth > 0)
    {
        Hittable::HitRecord record;
        switch (world.getCollisionData(ray, record, .001))
        {
        case Hittable::HitType::NO_HIT:
            color = background;
            break;
        case Hittable::HitType::HIT_NO_SCATTER:
            color = record.emitted;
            break;
        case Hittable::HitType::HIT_SCATTER:
            color = record.emitted + record.attenuation * computeRayColor(record.scatteredRay, background, world, depth - 1);
            // if (color.x() != 0. && color.x() != .73 && color.x() != .12)
            // {
            //     std::cerr << color.x() << ", " << color.y() << ", " << color.z() << std::endl;
            // }
            break;
        }
    }

    return color;
}

HittableList generatePerlinSpheres()
{
    HittableList objects;

    auto pertext = std::make_shared<TurbulentTexture>();
    objects.add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000., std::make_shared<LambertianDiffuse>(pertext)));
    objects.add(std::make_shared<Sphere>(Vec3{0., 2., 0.}, 2., std::make_shared<LambertianDiffuse>(pertext)));

    return objects;
}

HittableList generateImageTextureScene()
{
    HittableList objects;

    auto imgtext = std::make_shared<ImageTexture>("world.jpg");
    objects.add(std::make_shared<Sphere>(Vec3{0., 0., 0.}, 2., std::make_shared<LambertianDiffuse>(imgtext)));

    return objects;
}

HittableList simpleLightScene()
{
    HittableList objects;
    auto difflight = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(4., 4., 4.));
    auto mat = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(1., 0., 0.));

    objects.add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000., mat));
    objects.add(std::make_shared<Sphere>(Vec3{0., 2., 0.}, 2., mat));

    objects.add(std::make_shared<Sphere>(Vec3{0., 7., 0.}, 2., difflight));
    objects.add(std::make_shared<AARect<utils::Axis::Z>>(3., 5., 1., 3., -2., difflight));

    return objects;
}

HittableList cornellBox()
{
    HittableList objects;

    auto red = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.65, .05, .05));
    auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
    auto green = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.12, .45, .15));
    auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(15., 15., 15.));

    objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green));

    objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., light));

    objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white));

    objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));

    objects.add(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white));

    std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, white);

    box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
    box1 = std::make_shared<Translate>(box1, Vec3{265., 0., 295.});

    std::shared_ptr<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

    box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
    box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

    objects.add(box1);
    objects.add(box2);

    // objects.add(std::make_shared<Box>(Vec3{130., 0., 65.}, Vec3{295., 165., 230.}, white));
    // objects.add(std::make_shared<Box>(Vec3{265., 0., 295.}, Vec3{430., 330., 460.}, white));

    return objects;
}

void outputSphereScene(const int width, const int height, const int samplesPerPixel, const int maxReflections)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    Camera camera{static_cast<double>(width) / static_cast<double>(height), 40., 0., 10., Vec3{278., 278., -800.}, Vec3{278., 278., 0.}, 0., 1.};

    // Hittable::HitRecord record = {0};
    Vec3 pixelColor;
    HittableList world;
    Vec3 randomCenter0{0., .2, 0.};
    Vec3 background{0., 0., 0.};
    Vec3 randomCenter1;
    world = cornellBox();
    // double chooseMaterial;

    // auto checker = std::make_shared<CheckerTexture>(
    //     std::make_shared<SolidColor>(0.2, 0.3, 0.1),
    //     std::make_shared<SolidColor>(0.9, 0.9, 0.9), Vec3{10., 10., 10.});

    // world.add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000.,
    //                                    std::make_shared<LambertianDiffuse>(checker)));

    // for (int a = -11; a < 11; ++a)
    // {
    //     for (int b = -11; b < 11; ++b)
    //     {
    //         randomCenter0[0] = a + .9 * random_double();
    //         randomCenter0[2] = b + .9 * random_double();
    //         chooseMaterial = random_double();

    //         if ((randomCenter0 - Vec3{4., .2, 0.}).sqLen() > .81)
    //         {
    //             if (chooseMaterial < .7)
    //             {
    //                 randomCenter1 = randomCenter0;
    //                 randomCenter1[1] = random_double(0., .5);
    //                 world.add(std::make_shared<Sphere>(
    //                     randomCenter0, randomCenter1, .2,
    //                     std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(random_color())), 0., 1.));
    //             }
    //             else if (chooseMaterial < .9)
    //             {
    //                 world.add(std::make_shared<Sphere>(
    //                     randomCenter0, .2, std::make_shared<Metal>(std::make_shared<SolidColor>(random_color(Vec3{.5, .5, .5})), random_double(0., .5))));
    //             }
    //             else
    //             {
    //                 world.add(std::make_shared<Sphere>(
    //                     randomCenter0, .2, std::make_shared<Dielectric>(1.5)));
    //             }
    //         }
    //     }
    // }

    // world.add(std::make_shared<Sphere>(Vec3{0., 1., 0.}, 1., std::make_shared<Dielectric>(1.5)));
    // world.add(std::make_shared<Sphere>(Vec3{-4., 1., 0.}, 1., std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.4, .2, .1))));
    // world.add(std::make_shared<Sphere>(Vec3{4., 1., 0.}, 1., std::make_shared<Metal>(std::make_shared<SolidColor>(.7, .6, .5), 0.)));

    for (int i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int j = 0; j < width; ++j)
        {
            pixelColor.zero();
            for (int sample = 0; sample < samplesPerPixel; ++sample)
            {
                pixelColor += computeRayColor(camera.updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                              background, world, maxReflections);
            }
            pixelColor.formatColor(std::cout, samplesPerPixel);
        }
    }
    std::cerr << std::endl;
}

int main()
{
    outputSphereScene(600, 600, 100, 50);
}