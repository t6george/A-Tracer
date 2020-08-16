#include <iostream>
#include <memory>
#include <vector>

#include <SceneGeneration.cuh>
#include <SceneGeneration.hpp>
#include <Scenes.hpp>
#include <Pdfs.hpp>
#include <Objects.hpp>
#include <Material.hpp>

namespace cudagenerate
{
    __global__
    void scene(float * image, const unsigned int width, const unsigned int height, const unsigned int maxReflections)
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


    void launch_cuda_kernel(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel, 
		    const unsigned int maxReflections, const double aspectR)
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
	
	scene <<<blocks, threads>>> (d_img, width, height, maxReflections);

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
} // namespace generate
