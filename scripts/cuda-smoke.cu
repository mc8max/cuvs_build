// Small local-build / remote-execution check; no GPU required during compilation.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void answer(int* output) { *output = 42; }

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t error = (call);                                                    \
    if (error != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error));           \
      return 1;                                                                   \
    }                                                                             \
  } while (0)

int main()
{
  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
  int* output = nullptr;
  int result = 0;
  CUDA_CHECK(cudaMalloc(&output, sizeof(int)));
  answer<<<1, 1>>>(output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(output));
  std::printf("GPU: %s; SM %d.%d; kernel result: %d\n", device.name, device.major,
              device.minor, result);
  return result == 42 ? 0 : 1;
}
