#pragma once
#include <SceneRepresentation.hpp>

#define SCENE(name) HittableList name()

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene