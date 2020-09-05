#include <SceneGeneration.cuh>

int main()
{
    const double aspectR = 1.;
    const unsigned width = 500;
    const unsigned height = static_cast<int>(width / aspectR);
    const unsigned maxDepth = 50;

    generate::scene(width, height, maxDepth, aspectR);
}
