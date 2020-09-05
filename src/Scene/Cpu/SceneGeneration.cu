#if GPU == 0

#include <iostream>
#include <SharedPointer.cuh>
#include <vector>

#include <Macro.cuh>
#include <SceneGeneration.cuh>
#include <Scenes.cuh>
#include <Pdfs.cuh>
#include <Objects.cuh>
#include <Material.cuh>

namespace generate
{
    void ray_color(Ray &ray, const Vec3 &background, SharedPointer<HittableList> world, 
        WeightedPdf& pdf, const unsigned maxReflections, Vec3 &finalColor)
    {
        Vec3 color;
        Vec3 coeff {1., 1., 1.};
        Hittable::HitRecord record;
        bool active = true;

        for (unsigned reflections = 0; active && reflections < maxReflections; ++reflections)
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

    void scene(const unsigned width, const unsigned height, 
        const unsigned maxReflections, const double aspectR)
    {
        std::cout << "P3\n"
                << width << ' ' << height << "\n255\n";

        Vec3 pixelColor, tmp;
        SharedPointer<Camera> camera = nullptr;
        SharedPointer<HittableList> sampleObjects = nullptr;
        SharedPointer<HittableList> world = nullptr;
        Vec3 background;
	
	scene::cornell_box(camera, sampleObjects, world, background, aspectR);

        WeightedPdf pdf{stdSharedPointer::makeShared<CosinePdf>(), 
            stdSharedPointer::makeShared<HittablePdf>(sampleObjects), .5};

        for (int i = static_cast<int>(height) - 1; i >= 0; --i)
        {
            std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
            for (unsigned j = 0; j < width; ++j)
            {
                pixelColor.zero();
                for (unsigned sample = 0; sample < SAMPLES_PER_PIXEL; ++sample)
                {
                    generate::ray_color(camera->updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                 background, world, pdf, maxReflections, tmp);
		    pixelColor += tmp;
                }
                pixelColor.formatColor(std::cout, SAMPLES_PER_PIXEL);
            }
        }
        std::cerr << std::endl;
    }
} // namespace generate

#endif
