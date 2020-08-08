#pragma once

#include <memory>

#include <HittableList.hpp>
#include <Vec3.hpp>
#include <Ray.hpp>

namespace scene
{
    Vec3 compute_ray_color(Ray &ray, const Vec3 &background, HittableList &world, 
    	std::shared_ptr<HittableList> sampleObjects, const unsigned int bounceLimit);
    
    void generate(const unsigned int width, const unsigned int height, const unsigned int samplesPerPixel,
	const unsigned int maxReflections, const double aspectR);
} // namespace scene
