#pragma once

namespace cutils
{
    __device__ compute_ray_color(Ray &ray, const Vec3 &background, HittableList &world,
        std::shared_ptr<HittableList> sampleObjects, const int bounceLimit);
    __global__ generateScene(const int width, const int height, const int samplesPerPixel,
        const int maxReflections, const double aspectR);
} // namespace cutils
