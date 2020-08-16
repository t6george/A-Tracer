#pragma once

#include <memory>

#include <HittableList.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>

namespace generate
{
    __device__
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world,
        WeightedPdf& pdf, const unsigned int maxReflections, Vec3 &finalColor);

    __global__
    void scene(float * image, const unsigned int width, const unsigned int height, const unsigned int maxReflections);
    
    __global__
    void launch_cuda_kernel(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel,
                const unsigned int maxReflections, const double aspectR);
} // namespace generate
