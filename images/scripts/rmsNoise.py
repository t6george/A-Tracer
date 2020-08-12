import argparse
import numpy as np
from numba import cuda

PPM_HEADER_SIZE = 3

parser = argparse.ArgumentParser(description='Script to calculate the Root Mean Squared noise of a ppm image.')
parser.add_argument('-file', action="store", dest="fname", type=str)

# Signal definition taken from https://www.imatest.com/docs/noise/
@cuda.jit(device=True)
def evaluateSignal(r, g, b):
    return 0.2125 * r + 0.7154 * g + 0.0721 * b

@cuda.jit
def buildImageSignal(rComponent, gComponent, bComponent, signal):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, signal.shape[0], stride):
        signal[i] = evaluateSignal(rComponent[i], gComponent[i], bComponent[i])
        
if __name__ == '__main__':
    args = parser.parse_args()
    raw = []
    with open(args.fname, 'r') as file:
        raw = file.readlines()
        
    rComponent = np.empty(shape=(len(raw) - PPM_HEADER_SIZE), dtype=np.uint8)
    gComponent = np.empty_like(rComponent)
    bComponent = np.empty_like(rComponent)

    for i in range(PPM_HEADER_SIZE, len(raw)):
        components = raw[i].split()
        
        rComponent[i - PPM_HEADER_SIZE] = int(components[0])
        gComponent[i - PPM_HEADER_SIZE] = int(components[1])
        bComponent[i - PPM_HEADER_SIZE] = int(components[2])

    d_signal = cuda.device_array_like(rComponent)

    threads = 64
    blocks = d_signal.shape[0] // (threads * 4)

    buildImageSignal[blocks, threads](cuda.to_device(rComponent), cuda.to_device(gComponent), cuda.to_device(bComponent), d_signal)
    
    print('RMS Noise:', np.std(d_signal.copy_to_host()))
    
