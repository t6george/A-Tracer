#include <iostream>
#include <cstdint>

#include <Vec3.hpp>

void outputPPM(const uint16_t width, const uint16_t height)
{
    Vec3 colorV;
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";
    for (int32_t i = height - 1; i >= 0; --i)
    {
        std::cerr << "\rScanlines remaining: " << i << ' ' << std::flush;
        for (int32_t j = 0; j < width; ++j)
        {
            colorV[0] = static_cast<double>(j) / width;
            colorV[1] = static_cast<double>(i) / height;
            colorV[2] = .2;

            colorV.formatColor(std::cout);
        }
    }
    std::cerr << std::endl;
}

int main()
{
    outputPPM(200, 100);
}