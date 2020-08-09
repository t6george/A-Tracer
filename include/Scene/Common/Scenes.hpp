#pragma once
#include <utility>
#include <SceneRepresentation.hpp>

#define SCENE(name) std::pair<Camera, HittableList> name(const double aspectR)

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene