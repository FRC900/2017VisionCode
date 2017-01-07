#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#ifndef cudaSafeCall
#define cudaSafeCall(call,msg) cudaSafeCallWrapper((call),(msg),__FILE__,__LINE__)
#endif

#ifndef cublasSafeCall
#define cublasSafeCall(err, msg) cublasSafeCallWrapper(err, msg, __FILE__, __LINE__)
#endif

void cudaSafeCallWrapper(cudaError err, const char* msg, const char* file, const int line);
void cublasSafeCallWrapper(cublasStatus_t err, const char *msg, const char *file, const int line);




