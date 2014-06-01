import numpy as np
import math
import time
from numba import *
from numbapro import cuda
from numbapro.cudalib import curand
#import logging; logging.getLogger().setLevel(0)


RISKFREE = 0.02
VOLATILITY = 0.30


A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438

@cuda.jit(argtypes=(float32,), restype=float32, device=True, inline=True)
def cnd_cuda(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit(argtypes=(float32[:], float32[:], float32[:], float32[:], float32[:],
                    float32, float32))
def black_scholes_cuda(callResult, putResult, S, X,
                       T, R, V):
#    S = stockPrice
#    X = optionStrike
#    T = optionYears
#    R = Riskfree
#    V = Volatility
    i = cuda.grid(1)

    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd_cuda(d1)
    cndd2 = cnd_cuda(d2)

    expRT = math.exp((-1. * R) * T[i])
    callResult[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
    putResult[i] = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))

@cuda.jit(argtypes=(float32[:], float32, float32))
def c_distribute(rands, low, high):
    i = cuda.grid(1)

    if i >= rands.shape[0]:
        return

    rands[i] = (1.0 - rands[i]) * low + rands[i] * high


def main (*args):
    OPT_N = 4000000
    iterations = 10

    if len(args) >= 2:
        iterations = int(args[0])


    blockdim = 1024, 1
    griddim = int(math.ceil(float(OPT_N)/blockdim[0])), 1

    # Use cuRand to generate random numbers directyl on the gpu
    # to avoid memory transfers.
    prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)

    time0 = time.time()

    # malloc
    d_stockPrice = cuda.device_array(shape=(OPT_N), dtype=np.float32)
    d_optionStrike = cuda.device_array(shape=(OPT_N), dtype=np.float32)
    d_optionYears = cuda.device_array(shape=(OPT_N), dtype=np.float32)

    # Base distribution
    prng.uniform(d_stockPrice)
    prng.uniform(d_optionStrike)
    prng.uniform(d_optionYears)

    stream = cuda.stream()

    cfg_distribute = c_distribute[griddim, blockdim, stream]

    cfg_distribute(d_stockPrice, 5.0, 30.0)
    cfg_distribute(d_optionStrike, 1.0, 100.0)
    cfg_distribute(d_optionYears, 0.25, 10.)

    stream.synchronize()

    callResultNumbapro = np.zeros(OPT_N)
    putResultNumbapro = -np.ones(OPT_N)

    d_callResult = cuda.to_device(callResultNumbapro, stream)
    d_putResult = cuda.to_device(putResultNumbapro, stream)

    time1 = time.time()

    # Preconfigure the kernel as it's called multiple times in a loop.
    cfg_black_scholes_cuda = black_scholes_cuda[griddim, blockdim, stream]

    for i in range(iterations):
        cfg_black_scholes_cuda(
            d_callResult, d_putResult, d_stockPrice, d_optionStrike,
            d_optionYears, RISKFREE, VOLATILITY)

        d_callResult.to_host(stream)
        d_putResult.to_host(stream)

        stream.synchronize()

    time2 = time.time()
    dt = (time1 - time0) * 10 + (time2 - time1)

    print("numbapro.cuda time: %f msec" % ((1000 * dt) / iterations))

if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
