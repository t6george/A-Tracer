#pragma once

#include <Macro.cuh>
#include <SceneRepresentation.cuh>

#define SCENE(name) DEV void name(SharedPointer<Camera> &camera, SharedPointer<HittableList> &sampleObjects, SharedPointer<HittableList> &objects, Vec3 &bg, const double aspectR)

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene
