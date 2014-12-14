# Laplace 2D

> CHANGE: Since `numba` gained cuda support in 0.13 all examples
> in this directory can now be used without `numbapro`.

Implements jacobian relaxation on 4096x4096 matrices.

- laplace2d.py: naive implementation; very slow.
- laplace2d-numba.py: numba implementation.
- laplace2d-numbapro-gpu.py: naive numbapro cuda implementation.
- laplace2d-numbapro-gpu-smem.py: shared memory version of cuda implementation.
- laplace2d-numbapro-gpu-improve.py: shared memory + inline reduction on cuda.

Each script is runnable.


## Results

Machine:

* i7 4770k
* 16 GB DDR3 Memory
* GTX 760 2GB

### laplace2d

```
0, 0.250000 (elapsed: 42.418750 s)
...
```

### laplace2d-numba

```
0, 0.250000 (elapsed: 0.258061 s)
100, 0.002397 (elapsed: 7.603066 s)
200, 0.001204 (elapsed: 14.916623 s)
300, 0.000804 (elapsed: 22.252861 s)
400, 0.000603 (elapsed: 29.566275 s)
500, 0.000483 (elapsed: 36.912299 s)
600, 0.000403 (elapsed: 44.330453 s)
700, 0.000345 (elapsed: 51.820816 s)
800, 0.000302 (elapsed: 59.392504 s)
900, 0.000269 (elapsed: 66.991448 s)
total: 74.481207 s
```

### laplace2d-numbapro-gpu

```
0, 0.250000 (elapsed: 0.263038 s)
100, 0.002397 (elapsed: 10.419375 s)
200, 0.001204 (elapsed: 20.579178 s)
300, 0.000804 (elapsed: 30.675826 s)
400, 0.000603 (elapsed: 40.785381 s)
500, 0.000483 (elapsed: 50.931327 s)
600, 0.000403 (elapsed: 61.107004 s)
700, 0.000345 (elapsed: 71.402027 s)
800, 0.000302 (elapsed: 81.548966 s)
900, 0.000269 (elapsed: 91.769122 s)
total: 101.783953 s
```

### laplace2d-numbapro-gpu-smem

```
0, 0.250000 (elapsed: 0.259407 s)
100, 0.002397 (elapsed: 10.377766 s)
200, 0.001204 (elapsed: 20.566947 s)
300, 0.000804 (elapsed: 30.741831 s)
400, 0.000603 (elapsed: 40.925931 s)
500, 0.000483 (elapsed: 51.109859 s)
600, 0.000403 (elapsed: 61.369541 s)
700, 0.000345 (elapsed: 71.619761 s)
800, 0.000302 (elapsed: 81.875115 s)
900, 0.000269 (elapsed: 92.122988 s)
total: 102.370241 s
```

### laplace2d-numbapro-gpu-improve

```
0, 0.250000 (elapsed: 0.112200 s)
100, 0.002397 (elapsed: 1.569862 s)
200, 0.001204 (elapsed: 3.005556 s)
300, 0.000804 (elapsed: 4.442990 s)
400, 0.000603 (elapsed: 5.881128 s)
500, 0.000483 (elapsed: 7.319682 s)
600, 0.000403 (elapsed: 8.757301 s)
700, 0.000345 (elapsed: 10.193163 s)
800, 0.000302 (elapsed: 11.628956 s)
900, 0.000269 (elapsed: 13.062987 s)
total: 14.483667 s
```


## Proofiling

```bash
$ nvprof python laplace2d/laplace2d-numbapro-gpu-smem.py
Vendor:  Continuum Analytics, Inc.
Package: mkl
Message: trial mode expires in 30 days
==8777== NVPROF is profiling process 8777, command: python laplace2d/laplace2d-numbapro-gpu-smem.py
Vendor:  Continuum Analytics, Inc.
Package: numbapro
Message: trial mode expires in 30 days
Jacobi relaxation Calculation: 4096 x 4096 mesh
0, 0.250000 (elapsed: 0.263479 s)
100, 0.002397 (elapsed: 10.419311 s)
200, 0.001204 (elapsed: 20.538503 s)
300, 0.000804 (elapsed: 30.641278 s)
400, 0.000603 (elapsed: 40.950950 s)
500, 0.000483 (elapsed: 51.079881 s)
600, 0.000403 (elapsed: 61.295926 s)
700, 0.000345 (elapsed: 71.436986 s)
800, 0.000302 (elapsed: 81.624862 s)
900, 0.000269 (elapsed: 91.834681 s)
total: 101.840910 s
==8777== Profiling application: python laplace2d/laplace2d-numbapro-gpu-smem.py
==8777== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
65.44%  21.3086s      1000  21.309ms  20.937ms  26.951ms  [CUDA memcpy DtoH]
34.22%  11.1441s      1000  11.144ms  11.135ms  11.153ms  _cudapy_wrapper_jocabi_relax_core
0.34%  110.76ms         6  18.460ms  1.0240us  45.772ms  [CUDA memcpy HtoD]

==8777== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
99.42%  32.5733s      1000  32.573ms  32.181ms  38.247ms  cuMemcpyDtoHAsync
0.34%  111.40ms         6  18.566ms  11.560us  46.033ms  cuMemcpyHtoDAsync
0.13%  42.288ms         1  42.288ms  42.288ms  42.288ms  cuCtxCreate
0.09%  29.518ms      1000  29.517us  25.774us  98.938us  cuLaunchKernel
0.01%  3.0797ms         1  3.0797ms  3.0797ms  3.0797ms  cuModuleLoadDataEx
0.01%  2.3369ms      1000  2.3360us  1.7650us  11.195us  cuStreamSynchronize
0.00%  384.82us         4  96.204us  34.084us  125.26us  cuMemAlloc
0.00%  369.34us         2  184.67us     602ns  368.74us  cuDeviceGetCount
0.00%  30.432us         1  30.432us  30.432us  30.432us  cuDeviceGetName
0.00%  26.556us         1  26.556us  26.556us  26.556us  cuStreamCreate
0.00%  1.2030us         1  1.2030us  1.2030us  1.2030us  cuModuleGetFunction
0.00%     715ns         2     357ns     240ns     475ns  cuDeviceGet
0.00%     644ns         1     644ns     644ns     644ns  cuDeviceComputeCapability
0.00%     545ns         1     545ns     545ns     545ns  cuDeviceGetAttribute
```

```bash
$ nvprof python laplace2d/laplace2d-numbapro-gpu-improve.py
Vendor:  Continuum Analytics, Inc.
Package: mkl
Message: trial mode expires in 30 days
==8805== NVPROF is profiling process 8805, command: python laplace2d/laplace2d-numbapro-gpu-improve.py
Vendor:  Continuum Analytics, Inc.
Package: numbapro
Message: trial mode expires in 30 days
Jacobi relaxation Calculation: 4096 x 4096 mesh
0, 0.250000 (elapsed: 0.112581 s)
100, 0.002397 (elapsed: 1.533447 s)
200, 0.001204 (elapsed: 2.954463 s)
300, 0.000804 (elapsed: 4.374785 s)
400, 0.000603 (elapsed: 5.796029 s)
500, 0.000483 (elapsed: 7.216088 s)
600, 0.000403 (elapsed: 8.636411 s)
700, 0.000345 (elapsed: 10.057064 s)
800, 0.000302 (elapsed: 11.477560 s)
900, 0.000269 (elapsed: 12.898094 s)
total: 14.304708 s
==8805== Profiling application: python laplace2d/laplace2d-numbapro-gpu-improve.py
==8805== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
99.05%  13.9220s      1000  13.922ms  13.914ms  13.930ms  _cudapy_wrapper_jacobi_relax_core
0.64%  90.430ms         6  15.072ms     896ns  45.925ms  [CUDA memcpy HtoD]
0.31%  42.983ms      1000  42.983us  42.686us  58.398us  [CUDA memcpy DtoH]

==8805== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
98.88%  14.0259s      1000  14.026ms  13.986ms  14.692ms  cuMemcpyDtoHAsync
0.64%  91.088ms         6  15.181ms  11.510us  46.198ms  cuMemcpyHtoDAsync
0.30%  43.009ms         1  43.009ms  43.009ms  43.009ms  cuCtxCreate
0.09%  12.657ms         1  12.657ms  12.657ms  12.657ms  cuModuleLoadDataEx
0.07%  10.533ms      1000  10.532us  8.8490us  91.053us  cuLaunchKernel
0.01%  1.1934ms      1000  1.1930us  1.0260us  10.445us  cuStreamSynchronize
0.00%  375.45us         4  93.863us  34.034us  128.43us  cuMemAlloc
0.00%  372.45us         2  186.22us     542ns  371.90us  cuDeviceGetCount
0.00%  30.793us         1  30.793us  30.793us  30.793us  cuDeviceGetName
0.00%  26.357us         1  26.357us  26.357us  26.357us  cuStreamCreate
0.00%     963ns         1     963ns     963ns     963ns  cuModuleGetFunction
0.00%     789ns         2     394ns     260ns     529ns  cuDeviceGet
0.00%     630ns         1     630ns     630ns     630ns  cuDeviceComputeCapability
0.00%     517ns         1     517ns     517ns     517ns  cuDeviceGetAttributemaster
```
