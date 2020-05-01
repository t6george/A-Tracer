#include <iostream>
#include <cstdint>

void outputPPM(const uint16_t width, const uint16_t height)
{
    int r, g, b;
    std::cout << "P3\n"
              << width << ' ' << height << "\n255\n";
    for (int32_t i = height - 1; i >= 0; --i)
    {
        for (int32_t j = 0; j < width; ++j)
        {
            r = static_cast<int>((255.99 * j) / width);
            g = static_cast<int>((255.99 * i) / height);
            b = static_cast<int>(51.198);

            std::cout << r << ' ' << g << ' ' << b << "\n";
        }
    }
}

int main()
{
    outputPPM(200, 100);
}