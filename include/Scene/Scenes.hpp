#pragma once

#include <Macro.hpp>
#include <SceneRepresentation.hpp>

#define SCENE(name) DEV void name(std::shared_ptr<Camera> &camera, std::shared_ptr<HittableList> &sampleObjects, std::shared_ptr<HittableList> &objects, Vec3 &bg, const double aspectR)

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene
