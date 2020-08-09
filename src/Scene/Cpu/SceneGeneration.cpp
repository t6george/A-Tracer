#include <iostream>
#include <memory>
#include <vector>
#include <utility>

#include <SceneGeneration.hpp>
#include <Scenes.hpp>
#include <Pdfs.hpp>
#include <Objects.hpp>
#include <Material.hpp>

namespace generate
{
    Vec3 ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world, 
        WeightedPdf& pdf, const unsigned int bounceLimit)
    {
        Vec3 color {};
        Vec3 coeff {1., 1., 1.};
        unsigned int bounces = 0;
        Hittable::HitRecord record;
        bool active = true;

        for (; active && bounces < bounceLimit; ++bounces)
        {
            record = { 0 };
            switch (world->getCollisionData(ray, record, pdf, .001))
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
                    pdf.getPdf1()->construct(record.normal);
                    pdf.getPdf2()->construct(record.scatteredRay.getOrigin());

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

        std::tuple<std::shared_ptr<Camera>, std::shared_ptr<HittableList>, std::shared_ptr<HittableList>> scene 
            = scene::cornell_box(aspectR);

        Vec3 pixelColor;
        std::shared_ptr<Camera> camera = std::get<0>(scene);
        std::shared_ptr<HittableList> sampleObjects = std::get<1>(scene);
        std::shared_ptr<HittableList> world = std::get<2>(scene);
        Vec3 background{0., 0., 0.};

        WeightedPdf pdf{std::make_shared<CosinePdf>(), 
            std::make_shared<HittablePdf>(sampleObjects), .5};

        for (int i = static_cast<int>(height) - 1; i >= 0; --i)
        {
            std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
            for (unsigned int j = 0; j < width; ++j)
            {
                pixelColor.zero();
                for (unsigned int sample = 0; sample < samplesPerPixel; ++sample)
                {
                    pixelColor += generate::ray_color(camera->updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                                background, world, pdf, maxReflections);
                }
                pixelColor.formatColor(std::cout, samplesPerPixel);
            }
        }
        std::cerr << std::endl;
}
} // namespace scene
