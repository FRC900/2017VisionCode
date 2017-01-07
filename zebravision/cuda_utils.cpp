#include <cstdio>
#include <cstdlib>
#include "cuda_utils.hpp"

void cudaSafeCallWrapper(cudaError err, const char* msg, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr,
				"CUDA error : %s\nFile: %s\nLine Number: %d\nReason: %s\n",
				msg, file, line, cudaGetErrorString(err));
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

void cublasSafeCallWrapper(cublasStatus_t err, const char *msg, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) 
	{
		fprintf(stderr, 
				"CUBLAS error : %s\nFile: %s\nLine Number: %d\nErrno: %d\n",
				msg, file, line, err);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

