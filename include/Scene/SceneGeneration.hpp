#pragma once

#include <memory>

#include <HittableList.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>

namespace generate
{
#if GPU
    __device__
#endif
    void ray_color(Ray &ray, const Vec3 &background, std::shared_ptr<HittableList> world,
        WeightedPdf& pdf, const unsigned int maxReflections, Vec3 &finalColor);
    
#if GPU
    __global__
#endif
    void sample_pixel(float * image, const unsigned int width, const unsigned int height, 
        const unsigned int maxReflections);
        
    void scene(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel, 
        const unsigned int maxReflections, const double aspectR);
} // namespace generate
