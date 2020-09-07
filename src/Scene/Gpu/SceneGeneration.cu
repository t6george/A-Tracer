#if GPU == 1

#include <iostream>
#include <Memory.cuh>

#include <SceneGeneration.cuh>
#include <Scenes.cuh>
#include <Pdfs.cuh>
#include <Objects.cuh>
#include <Material.cuh>

namespace generate
{
    DEV void ray_color(Ray &ray, const Vec3 &background, SharedPointer<HittableList> world, 
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

    GLBL void sample_pixel(float * image, const unsigned width, const unsigned height, const unsigned maxReflections, 
		    SharedPointer<Camera> camera, WeightedPdf &pdf, const Vec3 &background, SharedPointer<HittableList> world)
    {
        extern__shared__ float samples[SAMPLES_PER_PIXEL * 3];

        Vec3 finalColor;

        unsigned idx = threadIdx.x * 3;
	
        ray_color(camera->updateLineOfSight((blockIdx.x + utils::random_double()) / width, (blockIdx.y + utils::random_double()) / height),
                 background, world, pdf, maxReflections, finalColor);

        samples[idx] = finalColor.x();
        samples[idx + 1] = finalColor.y();
        samples[idx + 2] = finalColor.z();

        for (unsigned i = 2; i < blockDim.x; i <<= 1)
        {
            __syncthreads();

            if ((threadIdx.x & (i - 1)) == 0)
            {
                samples[idx] += samples[idx + i * 3];
                samples[idx + 1] += samples[idx + i * 3 + 1];
                samples[idx + 2] += samples[idx + i * 3 + 2];
            }
        }

        idx = blockIdx.x * width + blockIdx.y;

        __syncthreads();

        if (threadIdx.x == 0)
        {            
            image[idx] = samples[0] / blockDim.x;
            image[idx + 1] = samples[1] / blockDim.x;
            image[idx + 2] = samples[2] / blockDim.x;
        }
    }

    HOST void scene(const unsigned width, const unsigned height, const unsigned maxReflections, const double aspectR)
    {
        std::cout << "P3\n"
            << width << ' ' << height << "\n255\n";

        Vec3 pixelColor;
        SharedPointer<Camera> camera;
        SharedPointer<HittableList> sampleObjects;
        SharedPointer<HittableList> world;
        Vec3 background;
            
        scene::cornell_box(camera, sampleObjects, world, background, aspectR);
            
        WeightedPdf pdf{SharedPointer<Pdf>(new CosinePdf),
                SharedPointer<Pdf>(new HittablePdf(mem::dynamic_pointer_cast<Hittable>(sampleObjects))), .5};

        float *h_img = nullptr;
        float *d_img = nullptr;

        cudaMallocHost(&h_img, sizeof(double) * width * height * 3);
        cudaMalloc(&d_img, sizeof(double) * width * height * 3);

        cudaMemcpy(d_img, h_img, width * height * 3, cudaMemcpyHostToDevice);

        dim3 threads = {SAMPLES_PER_PIXEL, 1, 1};
        dim3 blocks = {width, height, 1};

        sample_pixel <<<blocks, threads>>> (d_img, width, height, maxReflections, camera, pdf, background, world);

        cudaDeviceSynchronize();

        cudaMemcpy(h_img, d_img, width * height * 3, cudaMemcpyDeviceToHost);

        int idx;
            
        for (int i = static_cast<int>(height) - 1; i >= 0; --i)
            {
                for (unsigned j = 0; j < width; ++j)
                {
                    idx = (i * width + j) * 3;
                    Vec3{h_img[idx], h_img[idx + 1], h_img[idx + 2]}.formatColor(std::cout, SAMPLES_PER_PIXEL);
                }
            }

        cudaFreeHost(h_img);
        cudaFree(d_img);
    }
} // namespace generate

#endif
