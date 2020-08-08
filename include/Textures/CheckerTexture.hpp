#pragma once

#include <Texture.hpp>
#include <Vec3.hpp>

class CheckerTexture : public Texture
{
    const std::shared_ptr<Texture> tex1, tex2;
    const Vec3 scale;

public:
    CheckerTexture(const std::shared_ptr<Texture> tex1,
                   const std::shared_ptr<Texture> tex2,
                   const double x, const double y, const double z);

    CheckerTexture(const std::shared_ptr<Texture> tex1,
                   const std::shared_ptr<Texture> tex2, const Vec3 &scale);
    ~CheckerTexture() noexcept = default;

    Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};