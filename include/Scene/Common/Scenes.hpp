#pragma once
#include <utility>
#include <SceneRepresentation.hpp>

#define SCENE(name) std::tuple<std::shared_ptr<Camera>, std::shared_ptr<HittableList>, std::shared_ptr<HittableList>> name(const double aspectR)

namespace scene
{
    SCENE(cornell_box);

    SCENE(volume_cornell_box);

    SCENE(perlin_spheres);

    SCENE(image_texture);

    SCENE(light_scene);

    SCENE(summary);
} // namespace scene