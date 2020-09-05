#pragma once

#include <SharedPointer.cuh>

#include <HittableList.cuh>
#include <Camera.cuh>
#include <Vec3.cuh>
#include <Ray.cuh>

#define SAMPLES_PER_PIXEL 10

namespace generate
{
    DEV void ray_color(Ray &ray, const Vec3 &background, SharedPointer<HittableList> world,
        WeightedPdf& pdf, const unsigned int maxReflections, Vec3 &finalColor);
    
#ifdef __CUDACC__
    __global__
#endif
    void sample_pixel(float *image, const unsigned width, const unsigned height, const unsigned maxReflections, 
		    SharedPointer<Camera> camera, WeightedPdf &pdf, const Vec3 &background, SharedPointer<HittableList> world);
        
    void scene(const unsigned width, const unsigned height,
        const unsigned maxReflections, const double aspectR);
} // namespace generate
