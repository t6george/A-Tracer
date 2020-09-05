#pragma once

#include <SharedPointer.cuh>
#include <Texture.cuh>
#include <Vec3.cuh>

class CheckerTexture : public Texture
{
    const SharedPointer<Texture> tex1, tex2;
    const Vec3 scale;

public:
    HOST CheckerTexture(const SharedPointer<Texture> tex1,
                   const SharedPointer<Texture> tex2,
                   const double x, const double y, const double z);

    HOST CheckerTexture(const SharedPointer<Texture> tex1,
                   const SharedPointer<Texture> tex2, const Vec3 &scale);
   
    HOST ~CheckerTexture() noexcept = default;

    DEV Vec3 getValue(const double u, const double v, const Vec3 &point) const override;
};
