#pragma once
#include <memory>
#include <string>

#include <Texture.hpp>

class ImageTexture : public Texture
{
    const int bytesPerPixel = 3;
    int width, height, componentsPerPicture;
    std::unique_ptr<unsigned char> rgbData;
    int bytesPerScanline;

public:
    DEV HOST ImageTexture(const std::string &filename);
    DEV HOST ~ImageTexture() noexcept = default;

    DEV Vec3 getValue(double u, double v, const Vec3 &point) const override;
};