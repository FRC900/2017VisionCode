#include <iostream>
#include <cstdio>
#include "opencv2_3_shim.hpp"

#include "cuda_utils.hpp"

using std::cout;
using std::endl;
#if CV_MAJOR_VERSION == 2
using cv::gpu::PtrStepSz;
#elif CV_MAJOR_VERSION == 3
using cv::cuda::PtrStepSz;
#endif

// Take the output of the ZCA matrix mul - that will
// be a matrix. Each image is a row, each row is the pixels
// in BGRBGRBGR.. order
// Convert that to a flat 1-D array as expected by the neural
// net input stages
__global__ void split_image_channels(const PtrStepSz<float> input,
									 const size_t rows,
									 const size_t cols,
									 float * __restrict__ output)
{
	// 2D Index of current thread
	const int xIndex = threadIdx.x;
	const int yIndex = threadIdx.y;

	// Each image is its own block
	const int imgIndex = blockIdx.x;

	// Only valid threads perform memory I/O
	// See if thisis needed after setting the
	// block size to the image size
	if((xIndex < cols) && (yIndex < rows))
	{
		// yIndex * cols = number of floats per complete
		// filled row
		// add xIndex to get to the correct location in this row
		// Multiply by three to account for R, G, B float values
		//   per col in the input images
		const int flatIdxX = 3*(yIndex * cols + xIndex);
		const float blue   = input(imgIndex, flatIdxX + 0);
		const float green  = input(imgIndex, flatIdxX + 1);
		const float red	   = input(imgIndex, flatIdxX + 2);

		// Convert to flat 1-D representation
		// order is [image][color channel][row][col]
		const int chanDist = rows * cols;
		const int idx = imgIndex * 3 * chanDist + // 3 color channels of row*col pixels per image
			            yIndex * cols +         // select correct row   
						xIndex;                 // and the column in that row

		output[idx]                = blue;      // all the blue comes first
		output[idx +     chanDist] = green;     // then the green 
		output[idx + 2 * chanDist] = red;       // then the red from a given image
	}
}

// Math to add two intermediate steps of mean & stddev 
// See http://www.johndcook.com/blog/skewness_kurtosis/
__device__ void combine_running_totals(float &M1_1, const float M1_2, float &M2_1, const float M2_2, unsigned int &n_1, const unsigned int n_2)
{
	unsigned int combined_n = n_1 + n_2;

	const float delta  = M1_2 - M1_1;
	const float delta2 = delta * delta;

	float combined_M1 = (n_1 * M1_1 + n_2 * M1_2) / combined_n;
	float combined_M2 = M2_1 + M2_2 + delta2 * n_1 * n_2 / combined_n;

	n_1  = combined_n;
	M1_1 = combined_M1;
	M2_1 = combined_M2;
}

// For each input image, calculate the mean and stddev
// of each color channel in each image.  Then, for each
// pixel in a given image, apply global contrast normalization
// to the image - subtract the mean and divide by the stddev
// of the color channel of that image.
// input is an array of images, output is a 2d matrix where
// each image has been flattened into a single row
__global__ void mean_stddev_reduction_kernel(const PtrStepSz<float> * __restrict__ input,
												   PtrStepSz<float> output)
{
	// Thread index within block - used for addressing smem below
	const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

	// Shared memory per channel per thread = 2 floats for
	// mean and stddev sub-totals. Also keep 1 value per pixel
	// (i.e. 1 per each 3-channel set of above floats) for a total
	// of how many pixels have been processed.
	// So a 3 channel pixel needs 1 unsigned int and 6 floats
	// Thread blocks are up to 24x24 images, one thread per 3-channel pixel
	// TODO : fixme for variable sized thread blocks
	__shared__ float M1[24*24*3];
	__shared__ float M2[24*24*3];
	__shared__ unsigned int n[24*24];

	// 2D Index of current thread
	const int xIndex = threadIdx.x;
	const int yIndex = threadIdx.y;

	// Each image it its own block
	const int imgIndex = blockIdx.x;

	// TODO : cut the block size by 3x, do the first
	// reduction straight from memory?  Eliminates
	// 2/3rds of smem needed...

	// Will need do 3x the number of pixels in the final 
	// processing step at the end.

	// xIndex * 3 since col has a blue green and red component
	const float blue  = input[imgIndex](yIndex, 3*xIndex);
	const float green = input[imgIndex](yIndex, 3*xIndex + 1);
	const float red	  = input[imgIndex](yIndex, 3*xIndex + 2);

	// Initialize running average
	M1[tid * 3]     = blue;
	M1[tid * 3 + 1] = green;
	M1[tid * 3 + 2] = red;

	M2[tid * 3]     = 0;
	M2[tid * 3 + 1] = 0;
	M2[tid * 3 + 2] = 0;

	// Initialize pixel count
	n[tid] = 1;

    __syncthreads();

	// For each thread, combine the results from 2 or 3 threads
	// down into one. Each pass through the loop eliminates
	// half or 2/3rds of the partial results, eventually ending up
	// with just one final result per block

	// First do a pair of reductions by 3x. Both 12x12
	// and 24x24 have 3^2*2^N as prime factors, so this
	// will reduce the number of intermediate terms
	// to a power of 2.
    unsigned int s = (blockDim.x * blockDim.y) / 3;
	if (tid < s)
	{
		// N is the same for all 3 channels of a
		// given pixel. Re-use it when combining
		// the stats of the 3 channels
		unsigned int saved_n = n[tid];
		for (int i = 0; i < 3; i++)
		{
			// Blue, green, red = 3 entries per shared mem array
			const int i1 = 3 * tid + i;
			const int i2 = 3 * (tid + s) + i;
			n[tid] = saved_n;
			combine_running_totals(M1[i1], M1[i2], 
					M2[i1], M2[i2], 
					n[tid], n[tid + s]);
		}
		// N is the same for all 3 channels of a
		// given pixel. Re-use it when combining
		// the stats of the 3 channels
		saved_n = n[tid];
		for (int i = 0; i < 3; i++)
		{
			// Blue, green, red = 3 entries per shared mem array
			const int i1 = 3 * tid + i;
			const int i2 = 3 * (tid + 2*s) + i;
			n[tid] = saved_n;
			combine_running_totals(M1[i1], M1[i2], 
					M2[i1], M2[i2], 
					n[tid], n[tid + 2*s]);
		}
	}
	__syncthreads();
    s = (blockDim.x * blockDim.y) / 9;
	if (tid < s)
	{
		// N is the same for all 3 channels of a
		// given pixel. Re-use it when combining
		// the stats of the 3 channels
		unsigned int saved_n = n[tid];
		for (int i = 0; i < 3; i++)
		{
			// Blue, green, red = 3 entries per shared mem array
			const int i1 = 3 * tid + i;
			const int i2 = 3 * (tid + s) + i;
			n[tid] = saved_n;
			combine_running_totals(M1[i1], M1[i2], 
					M2[i1], M2[i2], 
					n[tid], n[tid + s]);
		}
		saved_n = n[tid];
		for (int i = 0; i < 3; i++)
		{
			// Blue, green, red = 3 entries per shared mem array
			const int i1 = 3 * tid + i;
			const int i2 = 3 * (tid + 2*s) + i;
			n[tid] = saved_n;
			combine_running_totals(M1[i1], M1[i2], 
					M2[i1], M2[i2], 
					n[tid], n[tid + 2*s]);
		}
	}
	__syncthreads();

	// At this point the number of intermediate
	// results is a power of two.  
	// Keep reducing by 2x until only one
	// result is left
	// At most 64 values are left, so stride
	// will be no more than 32. That fits in a
	// single warp.  
	// Instructions are SIMD synchronous in a warp
	//   so no need for syncthreads() anymore
	// Also, since there's memory allocated for 64 entries,
	// no point in checking for tid < s ... tid >= s will
	// be able to read and generate nonsense data that will
	// never be used, but that is quicker than a separate
	// condition check to keep them from running.

	if (tid < 32)
	{
		for (s = (blockDim.x * blockDim.y) / (3*3*2); s > 0; s >>= 1)
		{
			// N is the same for all 3 channels of a
			// given pixel. Re-use it when combining
			// the stats of the 3 channels
			unsigned int saved_n = n[tid];
			for (int i = 0; i < 3; i++)
			{
				// Blue, green, red = 3 entries per shared mem array
				const int i1 = 3 * tid + i;
				const int i2 = 3 * (tid + s) + i;
				n[tid] = saved_n;
				combine_running_totals(M1[i1], M1[i2], 
						M2[i1], M2[i2], 
						n[tid], n[tid + s]);
			}
		}
	}

	__syncthreads();

    // Update M1[0-2] and M2[0-2] with the 
    // mean and stddev of the B, G, R pixels
    if (tid < 3)
	{
		// M1 is the mean already - nothing extra needed

		// calculate stddev from M2 and n
		M2[tid] = sqrt(M2[tid] / n[0]);
	}
	__syncthreads();

	// Apply global contrast normalization to
	// each input image.
	// For each channel in each image, the mean and stddev has
	// already been calculated
	// For each channel in each pixel, subtract the mean and divide by the stddev
	// Insure only valid threads perform memory I/O
	// If the x/y index for this thread is beyond the
	// number of cols/rows, do nothing
	if((xIndex < input[imgIndex].cols) && (yIndex < input[imgIndex].rows))
	{
		// xIndex * 3 since col has a blue green and red component
		float blue	= input[imgIndex](yIndex, 3 * xIndex);
		float green	= input[imgIndex](yIndex, 3 * xIndex + 1);
		float red	= input[imgIndex](yIndex, 3 * xIndex + 2);

		blue  = (blue  - M1[0]) / M2[0];
		green = (green - M1[1]) / M2[1];
		red   = (red   - M1[2]) / M2[2];

		// yIndex * input[0].cols = number of floats per complete
		// filled row
		// add xIndex to get to the correct location in this row
		// Multiply by three to account for R, G, B float values
		//   per col in the input images
		const int flatIdxX = 3 * (yIndex * input[imgIndex].cols + xIndex);
		output(imgIndex, flatIdxX + 0) = blue;
		output(imgIndex, flatIdxX + 1) = green;
		output(imgIndex, flatIdxX + 2) = red;
	}
}

__host__ void cudaZCATransform(const std::vector<GpuMat> &input, 
		const GpuMat &weights, 
		PtrStepSz<float> *dPssIn,
		GpuMat &dFlattenedImages,
		GpuMat &zcaOut,
		float *output)
{
	// Create array of PtrStepSz entries corresponding to
	// each GPU mat in input. Copy it to device memory
	PtrStepSz<float> hPssIn[input.size()];
	for (size_t i = 0; i < input.size(); ++i)
		hPssIn[i] = input[i];
	cudaSafeCall(cudaMemcpy(dPssIn, hPssIn, input.size() * sizeof(PtrStepSz<float>), cudaMemcpyHostToDevice), "cudaMemcpy dPssIn");

	// Each block is one image
	const dim3 block(input[0].cols, input[0].rows);

	// x dimension is number of images
	const dim3 grid(input.size());

	// Todo : do this once in ZCA constructor
	// Create a CUDA stream. This lets us queue up a number of
	// cuda calls back to back and then later check to see
	// that they all finished
	cudaStream_t stream;
	cudaSafeCall(cudaStreamCreate(&stream), "ZCA cudaStreamCreate");

	// Todo : do this once in ZCA constructor
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate_v2(&handle), "cublasCreate");
    cublasSafeCall(cublasSetStream_v2(handle, stream), "cublasSetStream");

    cublasSafeCall(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");

	// Move these to device memory?
    const float alpha = 1.0;
    const float beta = 0.0;

	// Find the stddev and mean of each channel in each image
	// Then subtract the mean and divide by the stddev for each pixel
	mean_stddev_reduction_kernel<<<grid,block,0,stream>>>(dPssIn, dFlattenedImages);
	cudaSafeCall(cudaGetLastError(), "mean_stddev_reduction_kernel");

	// Multiply images by weights to get the ZCA-whitened output
	cublasSafeCall(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, weights.cols, dFlattenedImages.rows, weights.rows,
		&alpha,
		weights.ptr<float>(), static_cast<int>(weights.step / sizeof(float)),
		dFlattenedImages.ptr<float>(), static_cast<int>(dFlattenedImages.step / sizeof(float)),
		&beta,
		zcaOut.ptr<float>(), static_cast<int>(zcaOut.step / sizeof(float))),
		"cublasSgemm"	);

	// Copy to output buffer in the order expected by
	// neural net input
	split_image_channels<<<grid,block,0,stream>>>(zcaOut, input[0].rows, input[0].cols, output);
	cudaSafeCall(cudaGetLastError(), "split_image_channels kernel");

	cudaSafeCall(cudaStreamSynchronize(stream),"ZCA cudaStreamSynchronize failed");
	cublasSafeCall(cublasDestroy_v2(handle), "cublasDestroy");
	cudaSafeCall(cudaStreamDestroy(stream), "ZCA cudaStreamDestroy failed");
}
