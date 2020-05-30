#include <iostream>

#include <Utils.hpp>
#include <StbImageLibrary.hpp>
#include <ImageTexture.hpp>

ImageTexture::ImageTexture(const std::string &filename)
    : width{0}, height{0}, componentsPerPicture{bytesPerPixel},
      rgbData{stbi_load(filename.c_str(), &width, &height, &componentsPerPicture, componentsPerPicture)},
      bytesPerScanline{width * bytesPerPixel}
{
    if (!rgbData)
    {
        std::cerr << "Failed to load texture from file " << filename << std::endl;
    }
}

Vec3 ImageTexture::getValue(double u, double v, const Vec3 &point) const
{
    if (rgbData)
    {
        u = utils::clamp(u, 0., 1.);
        v = 1. - utils::clamp(v, 0., 1.);

        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);

        if (i >= width)
            i = width - 1;
        if (j >= height)
            j = height - 1;

        unsigned char *pixel = rgbData.get() + j * bytesPerScanline + i * bytesPerPixel;

        return Vec3{pixel[0] / 255., pixel[1] / 255., pixel[2] / 255.};
    }
    else
    {
        return Vec3{};
    }
}