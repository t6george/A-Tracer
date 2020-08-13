# Accelerated-Tracer
## Optimization Results
### Image Noise
Image noise is defined as the random variation in chroma (color) and/or luma (brightness). One way to measure noise is to compute the Root-Mean-Squared Noise by transforming the image into a 1D flat patch signal, and computing its standard deviation. This can be achieved by passing each pixel's RGB values through the reflectance formula https://docs.agi32.com/AGi32/Content/multi_use_forms/Reflectance_and_Color_Selection_-_Concepts.htm, which is designed to adjust the color's components according to human eye sensitivity. For the implementation, check out [images/scripts/rmsNoise.py](https://github.com/t6george/A-Tracer/blob/master/images/scripts/rmsNoise.py).

Unless the noisy, retro effect is desired, it is generally considered to be a contributing factor to an image's poor quality. That being said, the Monte-Carlo importance sampling approach I took reduced the image noise quite significantly. Consider the Cornell Box with Metallic, Lambertian and Dielectric objects below:

| ![](https://user-images.githubusercontent.com/31244240/90195415-4e333380-dd97-11ea-8a5b-c706b5c681f1.png)  | ![](https://user-images.githubusercontent.com/31244240/90195413-4d9a9d00-dd97-11ea-9c43-349f033e423e.png) |
|:---:|:---:|
| Uniform Sampling, 50.156 RMS Noise | Monte-Carlo Sampling, 41.991 RMS Noise

While the Monte-Carlo Image is rich with Salt and Pepper Noise (due to each pixel being an average of only 100 light ray samples), its RMS noise is 15.3% less than that of the image which reflects light rays in a uniformly random fashion.
