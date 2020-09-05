#include <Scenes.cuh>
#include <Materials.cuh>
#include <Objects.cuh>
#include <Textures.cuh>
#include <Transformations.cuh>
#include <Light.cuh>
#include <Pdfs.cuh>

namespace scene
{
    SCENE(cornell_box)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        auto red = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.65, .05, .05));
        auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
        auto green = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(15., 15., 15.));
        auto aluminum = std::make_shared<Metal>(std::make_shared<SolidColor>(.8, .85, .88), 0.);

        objects->add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green)));
        objects->add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
        objects->add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., light)));
        objects->add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white)));
        objects->add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));
        objects->add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white)));

        // SharedPointer<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, std::make_shared<Metal>(std::make_shared<SolidColor>(.8, .85, .88), 0.));
        SharedPointer<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, aluminum);

        box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
        box1 = std::make_shared<Translate>(box1, Vec3{300., 0., 295.});

        SharedPointer<Hittable> box3 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 400., 165.}, white);

        box3 = std::make_shared<AARotate<utils::Axis::Y>>(box3, -20.);
        box3 = std::make_shared<Translate>(box3, Vec3{90., 0., 295.});

        SharedPointer<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

        box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
        box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

        objects->add(box1);
        objects->add(box3);
        // objects->add(std::make_shared<Sphere>(Vec3{190., 390., 190.}, 90., red));
        objects->add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Dielectric>(1.5)));
        // objects->add(box2);

        sampleObjects->add(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., 
                std::make_shared<Material>(nullptr)));
        sampleObjects->add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Material>(nullptr)));
    }

    SCENE(perlin_spheres)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        auto pertext = std::make_shared<TurbulentTexture>();
        objects->add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000., std::make_shared<LambertianDiffuse>(pertext)));
        objects->add(std::make_shared<Sphere>(Vec3{0., 2., 0.}, 2., std::make_shared<LambertianDiffuse>(pertext)));
    }

    SCENE(image_texture)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        auto imgtext = std::make_shared<ImageTexture>("world.jpg");
        objects->add(std::make_shared<Sphere>(Vec3{0., 0., 0.}, 2., std::make_shared<LambertianDiffuse>(imgtext)));
    }

    SCENE(light_scene)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        auto difflight = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(4., 4., 4.));
        auto mat = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(1., 0., 0.));

        objects->add(std::make_shared<Sphere>(Vec3{0., -1000., 0.}, 1000., mat));
        objects->add(std::make_shared<Sphere>(Vec3{0., 2., 0.}, 2., mat));

        objects->add(std::make_shared<Sphere>(Vec3{0., 7., 0.}, 2., difflight));
        objects->add(std::make_shared<AARect<utils::Axis::Z>>(3., 5., 1., 3., -2., difflight));
    }

    SCENE(volume_cornell_box)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        auto red = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.65, .05, .05));
        auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
        auto green = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(15., 15., 15.));

        objects->add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green));

        objects->add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
        objects->add(std::make_shared<AARect<utils::Axis::Y>>(113., 443., 127., 432., 554., light));
        objects->add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white));
        objects->add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));
        objects->add(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white));

        SharedPointer<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, white);

        box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
        box1 = std::make_shared<Translate>(box1, Vec3{265., 0., 295.});

        SharedPointer<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

        box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
        box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

        objects->add(std::make_shared<ConstantVolume>(box1, std::make_shared<SolidColor>(0., 0., 0.), .01));
        objects->add(std::make_shared<ConstantVolume>(box2, std::make_shared<SolidColor>(1., 1., 1.), .01));
    }

    SCENE(summary)
    {
        const double fieldOfView = 40.;
        const double apertureRadius = 0.;
        const double distanceToFocus = 10.;
        const Vec3 lookFrom = Vec3{278., 278., -800.};
        const Vec3 lookAt = Vec3{278., 278., 0.};
        const double t0 = 0.;
        const double t1 = 1.;

        camera = std::make_shared<Camera>(aspectR, fieldOfView, apertureRadius, distanceToFocus, lookFrom, lookAt, t0, t1);

        SharedPointer<HittableList> boxes1 = std::make_shared<HittableList>();

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

                boxes1->add(std::make_shared<Box>(Vec3{x0, y0, z0}, Vec3{x1, y1, z1}, ground));
            }
        }

        objects = std::make_shared<HittableList>();
        sampleObjects = std::make_shared<HittableList>();

        objects->add(std::make_shared<BVHNode>(*boxes1.get(), 0., 1.));
        auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(7., 7., 7.));
        objects->add(std::make_shared<AARect<utils::Axis::Y>>(123., 423., 147., 412., 554., light));

        auto center1 = Vec3{400., 400., 200.};
        auto center2 = center1 + Vec3{30., 0., 0.};
        auto moving_sphere_material =
            std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.7, .3, .1));
        objects->add(std::make_shared<Sphere>(center1, center2, 50., moving_sphere_material));

        objects->add(std::make_shared<Sphere>(Vec3{260., 150., 45.}, 50., std::make_shared<Dielectric>(1.5)));
        objects->add(std::make_shared<Sphere>(
            Vec3{0., 150., 145.}, 50., std::make_shared<Metal>(std::make_shared<SolidColor>(Vec3{.8, .8, .9}), 10.)));

        auto boundary = std::make_shared<Sphere>(Vec3{360., 150., 145.}, 70., std::make_shared<Dielectric>(1.5));
        objects->add(boundary);
        objects->add(std::make_shared<ConstantVolume>(
            boundary, std::make_shared<SolidColor>(.2, .4, .9), .2));
        boundary = std::make_shared<Sphere>(Vec3{0., 0., 0.}, 5000., std::make_shared<Dielectric>(1.5));
        objects->add(std::make_shared<ConstantVolume>(
            boundary, std::make_shared<SolidColor>(1., 1., 1.), .0001));

        auto emat = std::make_shared<LambertianDiffuse>(std::make_shared<ImageTexture>("world.jpg"));
        objects->add(std::make_shared<Sphere>(Vec3{400., 200., 400.}, 100., emat));
        auto pertext = std::make_shared<PerlinNoiseTexture>(.1);
        objects->add(std::make_shared<Sphere>(Vec3{220, 280., 300.}, 80., std::make_shared<LambertianDiffuse>(pertext)));

        SharedPointer<HittableList> boxes2 = std::make_shared<HittableList>();
        
        auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
        int ns = 1000;
        for (int j = 0; j < ns; ++j)
        {
            boxes2->add(std::make_shared<Sphere>(Vec3::randomVector(Vec3{}, Vec3{165., 165., 165.}), 10., white));
        }

        objects->add(std::make_shared<Translate>(
            std::make_shared<AARotate<utils::Axis::Y>>(
                std::make_shared<BVHNode>(*boxes2.get(), 0., 1.), 15.),
            Vec3{-100., 270., 395.}));
    }

} // namespace scene
