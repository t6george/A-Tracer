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

    GLBL
    void sample_pixel(float * image, const unsigned width, const unsigned height, const unsigned maxReflections, 
		    SharedPointer<Camera> camera, WeightedPdf &pdf, const Vec3 &background, SharedPointer<HittableList> world)
    {
        /*extern*/__shared__ float samples[SAMPLES_PER_PIXEL * 3];

        Vec3 finalColor;
	
	//unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	
        //ray_color(camera->updateLineOfSight((x + utils::random_double()) / width, (y + utils::random_double()) / height),
        //          background, world, pdf, maxReflections, finalColor);

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

            unsigned idx = blockDim.y * blockIdx.y + blockIdx.x;
            
            image[idx] = samples[0] / blockDim.x;
            image[idx] = samples[1] / blockDim.x;
            image[idx] = samples[2] / blockDim.x;
        }
    }

    void scene(const unsigned width, const unsigned height, const unsigned maxReflections, const double aspectR)
    {
        std::cout << "P3\n"
            << width << ' ' << height << "\n255\n";

        Vec3 pixelColor;
        SharedPointer<Camera> camera;
        SharedPointer<HittableList> sampleObjects;
        SharedPointer<HittableList> world;
        Vec3 background;
            
        scene::cornell_box(camera, sampleObjects, world, background, aspectR);
            
        WeightedPdf pdf{mem::MakeShared<CosinePdf>(),
                mem::MakeShared<HittablePdf>(sampleObjects), .5};

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
