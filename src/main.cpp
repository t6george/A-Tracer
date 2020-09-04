#include <SceneGeneration.cuh>

int main()
{
    const double aspectR = 1.0;
    const unsigned width = 500;
    const unsigned height = static_cast<int>(width / aspectR);
    const unsigned samplesPerPixel = 10;
    const unsigned maxDepth = 50;

    generate::scene(width, height, samplesPerPixel, maxDepth, aspectR);
}
