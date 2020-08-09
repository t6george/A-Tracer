#include <iostream>
#include <SceneGeneration.cuh>

namespace generate
{
    __device__
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world,
        WeightedPdf& pdf, const int bounceLimit)
    {

    }

    __global__
    void scene(const int maxReflections, const double aspectR)
    {
	extern __shared__ char samples[blockDim.x];
    }

    void launch_cuda_kernel(const int width, const int height)
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
    }
} // namespace generate
