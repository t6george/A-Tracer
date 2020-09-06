#pragma once
#include <string>

#include <Memory.cuh>
#include <Texture.cuh>

class ImageTexture : public Texture
{
    const int bytesPerPixel = 3;
    int width, height, componentsPerPicture;
    UniquePointer<unsigned char> rgbData;
    int bytesPerScanline;

public:
    HOST ImageTexture(const std::string &filename);
    HOST ~ImageTexture() noexcept = default;

    DEV Vec3 getValue(double u, double v, const Vec3 &point) const override;
};
