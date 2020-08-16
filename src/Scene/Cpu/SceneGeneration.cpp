#include <iostream>
#include <memory>
#include <vector>

#include <SceneGeneration.hpp>
#include <Scenes.hpp>
#include <Pdfs.hpp>
#include <Objects.hpp>
#include <Material.hpp>

namespace generate
{
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world, 
        WeightedPdf& pdf, const unsigned int maxReflections, Vec3 &finalColor)
    {
        Vec3 color;
        Vec3 coeff {1., 1., 1.};
        Hittable::HitRecord record;
        bool active = true;

        for (unsigned int reflections = 0; active && reflections < maxReflections; ++reflections)
        {
            record = { 0 };
            switch (world->getCollisionData(ray, record, .001))
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
#if MONTE_CARLO
                    pdf.getPdf1()->construct(record.normal);
                    pdf.getPdf2()->construct(record.scatteredRay.getOrigin());
                    record.scatteredRay.setDirection(pdf.genRandomVector());
                    record.samplePdf = pdf.eval(record.scatteredRay.getDirection());
                    record.scatterPdf = fmax(0., record.normal.o(record.scatteredRay.getDirection().getUnitVector()) / utils::pi);
                    color += coeff * record.emitted;
                    coeff *= record.albedo * record.scatterPdf / record.samplePdf;
#else
                    record.scatteredRay.setDirection(Vec3::randomUnitHemisphereVec(record.normal));
                    color += coeff * record.emitted;
                    coeff *= record.albedo;
#endif
                }
                ray = record.scatteredRay;
                break;
            }
        }

        finalColor =  active ? Vec3{} : color;
    }

    void scene(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel, 
        const unsigned int maxReflections, const double aspectR)
    {
        std::cout << "P3\n"
                << width << ' ' << height << "\n255\n";

        Vec3 pixelColor, tmp;
        std::shared_ptr<Camera> camera = nullptr;
        std::shared_ptr<HittableList> sampleObjects = nullptr;
        std::shared_ptr<HittableList> world = nullptr;
        Vec3 background;
	
	scene::cornell_box(camera, sampleObjects, world, background, aspectR);

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
                    generate::ray_color(camera->updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                 background, world, pdf, maxReflections, tmp);
		    pixelColor += tmp;
                }
                pixelColor.formatColor(std::cout, samplesPerPixel);
            }
        }
        std::cerr << std::endl;
    }
} // namespace generate
