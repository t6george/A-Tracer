#pragma once

#ifdef __CUDACC__
    #define DEV __device__
    #define HOST __host__
#else
    #define DEV 
    #define HOST 
#endif
