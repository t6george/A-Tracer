#pragma once
#include <SceneRepresentation.hpp>

#if GPU
#define SCENE(name) __device__ void name(std::shared_ptr<Camera> &camera, std::shared_ptr<HittableList> &sampleObjects, std::shared_ptr<HittableList> &objects, Vec3 &
bg, const double aspectR)
#else
#define SCENE(name) void name(std::shared_ptr<Camera> &camera, std::shared_ptr<HittableList> &sampleObjects, std::shared_ptr<HittableList> &objects, Vec3 &bg, const double aspectR)
#endif

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene
