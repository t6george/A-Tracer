#include <iostream>
#include <memory>
#include <vector>

#if GPU
#include <SceneGeneration.cuh>
#else
#include <SceneGeneration.hpp>
#endif

#include <Scenes.hpp>
#include <Pdfs.hpp>
#include <Objects.hpp>
#include <Material.hpp>

namespace generate
{
#if GPU
    __device__
#endif
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
#if GPU
    {
        std::cout << "P3\n"
            << width << ' ' << height << "\n255\n";

            Vec3 pixelColor;
            std::shared_ptr<Camera> camera = nullptr;
            std::shared_ptr<HittableList> sampleObjects = nullptr;
            std::shared_ptr<HittableList> world = nullptr;
            Vec3 background;
            
        scene::cornell_box(camera, sampleObjects, world, background, aspectR);
            
        WeightedPdf pdf{std::make_shared<CosinePdf>(),
                std::make_shared<HittablePdf>(sampleObjects), .5};

        Vec3 *h_img = nullptr;
        Vec3 *d_img = nullptr;

        cudaMallocHost(&h_img, sizeof(double) * width * height * 3);
        cudaMalloc(&d_img, sizeof(double) * width * height * 3);

        cudaMemcpy(d_img, h_img, width * height * 3, cudaMemcpyHostToDevice);

        dim3 threads = {samplesPerPixel, 1, 1};
        dim3 blocks = {width, height, 1};

        sample_pixel <<<blocks, threads>>> (d_img, width, height, maxReflections);

        cudaDeviceSynchronize();

        cudaMemcpy(h_img, d_img, width * height * 3, cudaMemcpyDeviceToHost);

        int idx;
            
        for (int i = static_cast<int>(height) - 1; i >= 0; --i)
            {
                for (unsigned int j = 0; j < width; ++j)
                {
                    idx = (i * width + j) * 3;
                    Vec3{h_img[idx], h_img[idx + 1], h_img[idx + 2]}.formatColor(std::cout, samplesPerPixel);
                }
            }

        cudaFreeHost(h_img);
        cudaFree(d_img);
    }

    __global__
    void sample_pixel(float * image, const unsigned int width, const unsigned int height, 
        const unsigned int maxReflections)
    {
        /*extern*/__shared__ double samples[samplesPerPixel * 3];

        Vec3 finalColor;

        Vec3 color = generate::ray_color(camera->updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                                background, world, pdf, maxReflections, finalColor);

        samples[blockIdx.x * 3] = finalColor.x();
        samples[blockIdx.x * 3 + 1] = finalColor.y();
        samples[blockIdx.x * 3 + 2] = finalColor.z();

        __syncthreads();

        if (threadIdx.x == 0)
        {
            for (int i = 3; i < blockDim.x; i+=3)
            {
                samples[0] += samples[i];
                samples[1] += samples[i + 1];
                samples[2] += samples[i + 2];
            }

            int idx = blockDim.y * blockIdx.y + blockIdx.x;
            
            image[idx] = samples[0] / blockDim.x;
            image[idx] = samples[1] / blockDim.x;
            image[idx] = samples[2] / blockDim.x;
        }
    }
#else
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
#endif
} // namespace generate
