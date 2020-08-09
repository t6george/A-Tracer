#pragma once

#include <memory>

#include <HittableList.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>

namespace generate
{
    __device__
    Vec3 ray_color(Ray &ray, const Vec3 &background, HittableList &world,
        std::shared_ptr<HittableList> sampleObjects, const int bounceLimit);

    __global__
    void scene(const int width, const int height, const int samplesPerPixel,
        const int maxReflections, const double aspectR);
} // namespace generate
