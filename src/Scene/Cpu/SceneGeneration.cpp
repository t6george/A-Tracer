#include <iostream>
#include <SceneGeneration.hpp>

#include <memory>
#include <vector>

#include <Materials.hpp>
#include <Objects.hpp>
#include <Textures.hpp>
#include <Scene.hpp>
#include <Transformations.hpp>
#include <Light.hpp>
#include <Pdfs.hpp>
#include <Utils.hpp>


namespace generate
{
    HittableList cornellBox()
    {
        HittableList objects;

        auto red = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.65, .05, .05));
        auto white = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.73, .73, .73));
        auto green = std::make_shared<LambertianDiffuse>(std::make_shared<SolidColor>(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight>(std::make_shared<SolidColor>(15., 15., 15.));
        auto aluminum = std::make_shared<Metal>(std::make_shared<SolidColor>(.8, .85, .88), 0.);

        objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 555., green)));
        objects.add(std::make_shared<AARect<utils::Axis::X>>(0., 555., 0., 555., 0., red));
        objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(213., 343., 227., 332., 554., light)));
        objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 0., white)));
        objects.add(std::make_shared<AARect<utils::Axis::Y>>(0., 555., 0., 555., 555., white));
        objects.add(std::make_shared<FlipFace>(std::make_shared<AARect<utils::Axis::Z>>(0., 555., 0., 555., 555., white)));

        // std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, std::make_shared<Metal>(std::make_shared<SolidColor>(.8, .85, .88), 0.));
        std::shared_ptr<Hittable> box1 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 330., 165.}, aluminum);

        box1 = std::make_shared<AARotate<utils::Axis::Y>>(box1, 15.);
        box1 = std::make_shared<Translate>(box1, Vec3{300., 0., 295.});

        std::shared_ptr<Hittable> box3 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 400., 165.}, white);

        box3 = std::make_shared<AARotate<utils::Axis::Y>>(box3, -20.);
        box3 = std::make_shared<Translate>(box3, Vec3{90., 0., 295.});

        std::shared_ptr<Hittable> box2 = std::make_shared<Box>(Vec3{0., 0., 0.}, Vec3{165., 165., 165.}, white);

        box2 = std::make_shared<AARotate<utils::Axis::Y>>(box2, -18.);
        box2 = std::make_shared<Translate>(box2, Vec3{130., 0., 65.});

        objects.add(box1);
        objects.add(box3);
        // objects.add(std::make_shared<Sphere>(Vec3{190., 390., 190.}, 90., red));
        objects.add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Dielectric>(1.5)));
        // objects.add(box2);

        return objects;
    }

    Vec3 ray_color(Ray &ray, const Vec3 &background, HittableList &world, 
        std::shared_ptr<HittableList> sampleObjects, const unsigned int bounceLimit)
    {
        Vec3 color {};
        Vec3 coeff {1., 1., 1.};
        unsigned int bounces = 0;
        Hittable::HitRecord record;
        bool active = true;

        for (; active && bounces < bounceLimit; ++bounces)
        {
            record = { 0 };
            switch (world.getCollisionData(ray, record, .001))
            {
            case Hittable::HitType::NO_HIT:
                color += background * coeff;
                active = false;
                break;
            case Hittable::HitType::HIT_NO_SCATTER:
                color += record.emitted * coeff;
                active = false;
                break;
            case Hittable::HitType::HIT_SCATTER:
                if (record.isSpecular)
                {
                    coeff *= record.albedo; 
                }
                else
                {
                    WeightedPdf pdf{std::make_shared<CosinePdf>(record.normal), 
                        std::make_shared<HittablePdf>(sampleObjects, record.scatteredRay.getOrigin()), .5};

                    record.scatteredRay.setDirection(pdf.genRandomVector());
                    record.samplePdf = pdf.eval(record.scatteredRay.getDirection());
                    record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);

                    color += coeff * record.emitted;
                    coeff *= record.albedo * record.scatterPdf / record.samplePdf;
                }
                ray = record.scatteredRay;
                break;
            }
        }

        return active ? Vec3{} : color;
    }

    void scene(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel, const unsigned int maxReflections, const double aspectR)
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
        sampleObjects->add(std::make_shared<Sphere>(Vec3{190., 90., 190.}, 90., std::make_shared<Material>(nullptr)));

        for (int i = height - 1; i >= 0; --i)
        {
            std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
            for (int j = 0; j < width; ++j)
            {
                pixelColor.zero();
                for (int sample = 0; sample < samplesPerPixel; ++sample)
                {
                    pixelColor += generate::ray_color(camera.updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                                background, world, sampleObjects, maxReflections);
                }
                pixelColor.formatColor(std::cout, samplesPerPixel);
            }
        }
        std::cerr << std::endl;
}
} // namespace scene