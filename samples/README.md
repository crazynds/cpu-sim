# Samples CUDA

Programas CUDA de referência para o CudaSimulator. Cada um roda na GPU real, compara o resultado com uma implementação em CPU e imprime `PASS`/`FAIL`. Depois, os mesmos fontes servem de entrada para o nosso compilador: a saída precisa continuar `PASS`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

O device code é compilado com `-fmad=false` (opção `SAMPLES_NO_FMAD`), para que os resultados float sejam bit a bit iguais aos da CPU onde a matemática é IEEE exata.

## O que cada sample exercita

| Sample | Shared mem | `__syncthreads` | Divergência | Memória | Atômicas | Warp intrinsics |
|---|---|---|---|---|---|---|
| `matmul` | tiles 16x16 | dentro de loop | early return (naive) | contíguo + strided | | |
| `nbody` | tile `float4`, leitura uniforme (broadcast) | dentro de loop | cauda do último bloco | `float4` | | |
| `reduction` | árvore | dentro de loop | conjunto ativo encolhendo | grid-stride | float, endereço uniforme | `__shfl_down_sync` |
| `histogram` | 256 bins | init / merge | merge condicional | load de byte | int, muito conflito, shared e global | |
| `heat2d` | tile + halo | 1 por launch | halo só nas bordas do bloco, condição de contorno | clamp, blocos parciais 2D | | |
| `scan` | Blelloch | 1 por nível (18) | ativo cai pela metade e volta a dobrar | índices calculados em shared | | |
| `transpose` | tile com padding `[32][33]` | 1 | bordas | escrita strided (scatter) vs contígua | | |
| `bitonic_sort` | segmento de 1024 | loop aninhado (55 por bloco) | swap depende dos dados | índices por bits | | |
| `spmv` | | | trip count por lane (0 a 64) | gather real `x[col[k]]` | | `__shfl_down_sync` |
| `mandelbrot` | | | loop divergente, máscara encolhe lane a lane | | | |

## Runtime API usada

Só o mínimo, que é a primeira superfície que o nosso runtime vai precisar implementar:

`cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaGetLastError`, `cudaGetErrorString`

Recursos de linguagem usados: `__global__`, `__device__`, `__host__ __device__`, `__shared__` estático, `threadIdx/blockIdx/blockDim/gridDim` em 1D e 2D, `dim3`, `float3/float4`, `min/max`, `rsqrtf`, `atomicAdd` (int e float), `__shfl_down_sync`, `constexpr` usado no device.
