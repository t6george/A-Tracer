#include <SceneGeneration.cuh>

int main()
{
    const double aspectR = 1.0;
    int width = 500;
    int height = static_cast<int>(width / aspectR);
    int maxDepth = 50;

    generate::scene(width, height, maxDepth, aspectR);
}
