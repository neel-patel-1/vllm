#!/bin/python3
import cupy as cp
import ctypes

# Load the CUDA runtime library to access profiler start/stop APIs.
cudart = ctypes.CDLL("libcudart.so")

def cuda_profiler_start():
    """Start the CUDA profiler."""
    cudart.cudaProfilerStart()

def cuda_profiler_stop():
    """Stop the CUDA profiler."""
    cudart.cudaProfilerStop()

# Define a simple CUDA kernel that adds two arrays.
kernel_code = r'''
extern "C" __global__
void add_arrays(const float* a, const float* b, float* c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
'''

# Compile the kernel using CuPy's RawKernel.
add_kernel = cp.RawKernel(kernel_code, 'add_arrays')

# Set up data: allocate two input arrays and one output array.
N = 1 << 20  # 1 million elements
a = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)
c = cp.empty_like(a)

# Determine grid and block sizes.
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# Warm up the GPU: launch the kernel once before profiling.
add_kernel((blocks_per_grid,), (threads_per_block,), (a, b, c, N))
cp.cuda.Stream.null.synchronize()

# --- Start of the profiling range ---
cuda_profiler_start()

# Launch the kernel that we want to profile.
add_kernel((blocks_per_grid,), (threads_per_block,), (a, b, c, N))
cp.cuda.Stream.null.synchronize()

add_kernel((blocks_per_grid,), (threads_per_block,), (a, b, c, N))
cp.cuda.Stream.null.synchronize()

add_kernel((blocks_per_grid,), (threads_per_block,), (a, b, c, N))
cp.cuda.Stream.null.synchronize()

# --- End of the profiling range ---
cuda_profiler_stop()

print("Kernel execution completed.")