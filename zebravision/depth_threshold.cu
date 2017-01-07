#include <vector>
#include "cuda_utils.hpp"
#include "opencv2_3_shim.hpp"

#if CV_MAJOR_VERSION == 2
using cv::gpu::PtrStepSz;
#elif CV_MAJOR_VERSION == 3
using cv::cuda::PtrStepSz;
#endif

// Be conservative here - if any of the depth values in the
// target rect are in the expected range, consider the entire
// rect in range.  Also say that it is in range if any of the
// depth values are negative or NaN (i.e. no depth info for those
// pixels)
__device__ bool depth_test_logic(const float depth, 
								 const float depthMin,
								 const float depthMax)
{
	if (isnan(depth) || (depth <= 0.0) || 
		((depth > depthMin) && (depth < depthMax)))
		return true;
	return false;
}

// Given a depth map in input, see if any value is
// in the range between depthMin and depthMax.  If
// so, set answer to true. If all pixels fall outside
// the range, set answer to false.
__global__ void depth_threshold_kernel(const PtrStepSz<float> * __restrict__ input,
									   const float depthMin,
									   const float depthMax,
									   bool *answer)
{
	// Thread index within block - used for addressing smem below
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ bool inRange[12*4];

	// 2D pixel index of current thread
	const int xIndex = threadIdx.x;
	const int yIndex = threadIdx.y;

	const int imgIndex = blockIdx.x;

	// General approach is to populate shared mem with true/false
	// values from input pixels. Then OR together results from sets
	// of pixels iteratively until all results have been ORd into
	// inRange[0].
	// The input images are 12x12. That's not a power of two, so 
	// do two reductions by 3x before finishing up reducing by 2x four times.
	// Use initial read from memory to do the first 3x reduction. That
	// reduces the number of threads and shared memory needed by a factor of 3.

	// Be conservative here - if any of the depth values in the 
	// target rect are in the expected range, consider the entire 
	// rect in range.  Also say that it is in range if any of the 
	// depth values are negative (i.e. no depth info for those pixels)
	bool myinRange = 
		depth_test_logic(input[imgIndex](yIndex, xIndex), depthMin, depthMax) ||
		depth_test_logic(input[imgIndex](yIndex + blockDim.y, xIndex), depthMin, depthMax) ||
		depth_test_logic(input[imgIndex](yIndex + 2*blockDim.y, xIndex), depthMin, depthMax);

	inRange[tid] = myinRange;

	// Let all threads finish the compare and put
	// their results in shared mem
    __syncthreads();

	// Do another reduction by 3x here, from 
	// shared mem to shared mem
	// Simply OR the results together to propagate
	// true values down to inRange[0]
	const unsigned int s = (blockDim.x * blockDim.y) / 3;
	if (tid < s)
		inRange[tid] = myinRange = inRange[tid] | inRange[tid + s] | inRange[tid + 2*s];

    __syncthreads();

	if (tid < 8)
	{
#if 0
		myinRange |= inRange[tid + 8];
		myinRange |= inRange[tid + 4];
		myinRange |= inRange[tid + 2];
		myinRange |= inRange[tid + 1];
#else
		// Final set of reductions by 2x to get down to one result
		for (unsigned int s = (blockDim.x * blockDim.y) / 6; s > 0; s >>= 1)
		{
			// Basically just propagate any true values
			// down to thread 0 - only return false
			// if the entire set of compares was false
			//if (tid < s)
				inRange[tid] |= inRange[tid + s];
			//myinRange |= __shfl_down(myinRange, s);
		}
#endif
	}

    if (tid == 0)
		answer[imgIndex] = inRange[0];
}

__host__ std::vector<bool> cudaDepthThreshold(const std::vector<GpuMat> &depthList, const float depthMin, const float depthMax)
{
	// Create array of PtrStepSz entries corresponding to
	// each GPU mat in depthList. Copy it to device memory

	PtrStepSz<float> hPssIn[depthList.size()];
	for (size_t i = 0; i < depthList.size(); ++i)
		hPssIn[i] = depthList[i];
	PtrStepSz<float> *dPssIn;
	cudaSafeCall(cudaMalloc(&dPssIn, depthList.size() * sizeof(*dPssIn)), "cudaMalloc threshold dPssIn");
	cudaSafeCall(cudaMemcpy(dPssIn, hPssIn, depthList.size() * sizeof(PtrStepSz<float>), cudaMemcpyHostToDevice), "cudaMemcpy dPssIn");
	
	bool *dResult;
	cudaSafeCall(cudaMalloc(&dResult, depthList.size() * sizeof(bool)), "cudaMalloc threshold result");

	// Each block is one depth
	// Set the block size to the smallest power
	// of two large enough to hold an depth
	const dim3 block(12, 4);

	// each block is 1 image
	const dim3 grid(depthList.size());

	depth_threshold_kernel<<<grid, block>>>(dPssIn, depthMin, depthMax, dResult);
	cudaSafeCall(cudaDeviceSynchronize(), "depthThreshold cudaDeviceSynchronize failed");

	bool hResult[depthList.size()];
	cudaSafeCall(cudaMemcpy(&hResult, dResult, sizeof(bool) * depthList.size(), cudaMemcpyDeviceToHost), "cudaMemcpy depth result");
	cudaSafeCall(cudaFree(dPssIn), "depthThreshold cudaFree");
	cudaSafeCall(cudaFree(dResult), "depthThreshold cudaFree");

	return std::vector<bool>(hResult, hResult + depthList.size());
}
