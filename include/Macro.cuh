#pragma once

#ifdef __CUDACC__
    #define DEV __device__
    #define HOST __host__
    #define GLBL __global__
#else
    #define DEV 
    #define HOST 
    #define GLBL 
#endif
