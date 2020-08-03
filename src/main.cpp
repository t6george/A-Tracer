#include <iostream>
#include <memory>
#include <vector>

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
#include <ConstantVolume.hpp>
#include <BVHNode.hpp>
#include <HittablePdf.hpp>
#include <CosinePdf.hpp>
#include <WeightedPdf.hpp>
#include <Ray.hpp>
#include <HittableList.hpp>
#include <Sphere.hpp>
#include <Camera.hpp>
#include <LambertianDiffuse.hpp>
#include <Metal.hpp>
#include <Dielectric.hpp>

Vec3 computeRayColor(const Ray &ray, const Vec3 &background, HittableList &world, 
    std::shared_ptr<HittableList> sampleObjects, int depth)
{
    Vec3 color;
    if (depth > 0)
    {
        Hittable::HitRecord record = { 0 };
        switch (world.getCollisionData(ray, record, .001))
        {
        case Hittable::HitType::NO_HIT:
            color = background;
            break;
        case Hittable::HitType::HIT_NO_SCATTER:
            color = record.emitted;
            break;
        case Hittable::HitType::HIT_SCATTER:
            if (record.isSpecular)
            {
                color = record.albedo * 
                    computeRayColor(record.scatteredRay, background, world, sampleObjects, depth - 1);
            }
            else
            {
                WeightedPdf pdf{std::make_shared<CosinePdf>(record.normal), 
                    std::make_shared<HittablePdf>(sampleObjects, record.scatteredRay.getOrigin()), .5};

                record.scatteredRay.setDirection(pdf.genRandomVector());
                record.samplePdf = pdf.eval(record.scatteredRay.getDirection());
                record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);

                color = record.emitted + record.albedo * record.scatterPdf *
                        computeRayColor(record.scatteredRay, background, world, sampleObjects, depth - 1) / record.samplePdf;
            }
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

    objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green)));
    objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
    objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., light)));
    objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white)));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));
    objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white)));

    // std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, std::make_shared<Metal>(std::make_shared<SolidColor>(.8, .85, .88), 0.));
    std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, white);

    box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
    box1 = std::make_shared<Translate>(box1, Vec3{265., 0., 295.});

    std::shared_ptr<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

    box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
    box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

    objects.add(box1);
    objects.add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Dielectric>(1.5)));
    // objects.add(box2);

    return objects;
}

HittableList volumeCornellBox()
{
    HittableList objects;

    auto red = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.65, .05, .05));
    auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
    auto green = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.12, .45, .15));
    auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(15., 15., 15.));

    objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green));

    objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(113., 443., 127., 432., 554., light));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));
    objects.add(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white));

    std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, white);

    box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
    box1 = std::make_shared<Translate>(box1, Vec3{265., 0., 295.});

    std::shared_ptr<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

    box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
    box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

    objects.add(std::make_shared<ConstantVolume>(box1, std::make_shared<SolidColor>(0., 0., 0.), .01));
    objects.add(std::make_shared<ConstantVolume>(box2, std::make_shared<SolidColor>(1., 1., 1.), .01));

    return objects;
}

HittableList theNextWeekSummaryScene()
{
    HittableList boxes1;
    auto ground = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.48, .83, .53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; ++i)
    {
        for (int j = 0; j < boxes_per_side; ++j)
        {
            auto w = 100.;
            auto x0 = -1000. + i * w;
            auto z0 = -1000. + j * w;
            auto y0 = 0.;
            auto x1 = x0 + w;
            auto y1 = utils::random_double(1., 101.);
            auto z1 = z0 + w;

            boxes1.add(std::make_shared<Box>(Vec3{x0, y0, z0}, Vec3{x1, y1, z1}, ground));
        }
    }

    HittableList objects;

    objects.add(std::make_shared<BVHNode>(boxes1, 0., 1.));
    auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(7., 7., 7.));
    objects.add(std::make_shared<AARect<utils::Axis::Y>>(123., 423., 147., 412., 554., light));

    auto center1 = Vec3{400., 400., 200.};
    auto center2 = center1 + Vec3{30., 0., 0.};
    auto moving_sphere_material =
        std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.7, .3, .1));
    objects.add(std::make_shared<Sphere>(center1, center2, 50., moving_sphere_material));

    objects.add(std::make_shared<Sphere>(Vec3{260., 150., 45.}, 50., std::make_shared<Dielectric>(1.5)));
    objects.add(std::make_shared<Sphere>(
        Vec3{0., 150., 145.}, 50., std::make_shared<Metal>(std::make_shared<SolidColor>(Vec3{.8, .8, .9}), 10.)));

    auto boundary = std::make_shared<Sphere>(Vec3{360., 150., 145.}, 70., std::make_shared<Dielectric>(1.5));
    objects.add(boundary);
    objects.add(std::make_shared<ConstantVolume>(
        boundary, std::make_shared<SolidColor>(.2, .4, .9), .2));
    boundary = std::make_shared<Sphere>(Vec3{0., 0., 0.}, 5000., std::make_shared<Dielectric>(1.5));
    objects.add(std::make_shared<ConstantVolume>(
        boundary, std::make_shared<SolidColor>(1., 1., 1.), .0001));

    auto emat = std::make_shared<LambertianDiffuse>(std::make_shared<ImageTexture>("world.jpg"));
    objects.add(std::make_shared<Sphere>(Vec3{400., 200., 400.}, 100., emat));
    auto pertext = std::make_shared<PerlinNoiseTexture>(.1);
    objects.add(std::make_shared<Sphere>(Vec3{220, 280., 300.}, 80., std::make_shared<LambertianDiffuse>(pertext)));

    HittableList boxes2;
    auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; ++j)
    {
        boxes2.add(std::make_shared<Sphere>(Vec3::randomVector(Vec3{}, Vec3{165., 165., 165.}), 10., white));
    }

    objects.add(std::make_shared<Translate>(
        std::make_shared<AARotate<utils::Axis::Y>>(
            std::make_shared<BVHNode>(boxes2, 0., 1.), 15.),
        Vec3{-100., 270., 395.}));

    return objects;
}

void outputSphereScene(const int width, const int height, const int samplesPerPixel, const int maxReflections, const double aspectR)
{
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";

    const double fieldOfView = 40.;
    const double apertureRadius = 0.;
    const double distanceToFocus = 10.;
    const Vec3 lookFrom = Vec3{278., 278., -800.};
    const Vec3 lookAt = Vec3{278., 278., 0.};
    const double t0 = 0.;
    const double t1 = 1.;

    Camera camera{aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1};

    Vec3 pixelColor;
    HittableList world;
    Vec3 randomCenter0{0., .2, 0.};
    Vec3 background{0., 0., 0.};
    Vec3 randomCenter1;
    world = cornellBox();

    std::shared_ptr<HittableList> sampleObjects = std::make_shared<HittableList>();
    sampleObjects->add(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., 
            std::make_shared<Material>(nullptr)));
    //sampleObjects->add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Material>(nullptr)));

    for (int i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int j = 0; j < width; ++j)
        {
            pixelColor.zero();
            for (int sample = 0; sample < samplesPerPixel; ++sample)
            {
                pixelColor += computeRayColor(camera.updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                              background, world, sampleObjects, maxReflections);
            }
            pixelColor.formatColor(std::cout, samplesPerPixel);
        }
    }
    std::cerr << std::endl;
}

int main()
{
    const double aspectR = 1.0;
    int width = 500;
    int height = static_cast<int>(width / aspectR);
    int samplesPerPixel = 100;
    int maxDepth = 50;

    outputSphereScene(width, height, samplesPerPixel, maxDepth, aspectR);
}