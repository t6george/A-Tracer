#include <iostream>
#include <memory>
#include <vector>

#include <SceneGeneration.cuh>
#include <Scenes.hpp>
#include <Pdfs.hpp>
#include <Objects.hpp>
#include <Material.hpp>

<<<<<<< Updated upstream
namespace cudagenerate
{
    __device__
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world,
        WeightedPdf& pdf, const int bounceLimiti, Vec3 &finalColor)
=======
namespace generate
{
    __device__
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world,
        WeightedPdf& pdf, const int bounceLimit)
>>>>>>> Stashed changes
    {

    }

<<<<<<< Updated upstream

    __global__
    void scene(float * image, const unsigned int width, const unsigned int height)
    {
	/*extern*/__shared__ double samples[blockDim.x * 3];

        Vec3 color = cudagenerate::ray_color(camera->updateLineOfSight((j + utils::random_double()) / width, (i + utils::random_double()) / height),
                                                background, world, pdf, maxReflections);
	
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


    void launch_cuda_kernel(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel, const unsigned int maxReflections, const double aspectR)
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
	
	for (unsigned int i = 0; i < width * height; ++i)
	{
		h_img[i] = 0.;
	}
	
	cudaMemcpy(d_img, h_img, width * height);

	cudaDeviceSynchronize();
	
	int idx;
        
	for (int i = static_cast<int>(height) - 1; i >= 0; --i)
        {
            for (unsigned int j = 0; j < width; ++j)
            {
		idx = (i * width + j) * 3;
                Vec3{h_img[idx], h_img[idx + 1], h_img[idx + 2]}.formatColor(std::cout, samplesPerPixel);
            }
        }
    }
} // namespace generate
=======
    __global__
    void scene(const int width, const int height, const int samplesPerPixel,
        const int maxReflections, const double aspectR)
    {

    }
} // namespace generate
>>>>>>> Stashed changes
